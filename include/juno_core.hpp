#pragma once
#ifndef JUNO_CORE_H_
#define JUNO_CORE_H_

#include "utils.hpp"
#include "juno_rt.hpp"
#include "juno_query.hpp"
#include "juno_gpu_kernel.cuh"
namespace juno {

template <typename T>
class juno_core {
private:
    std::string dataset_dir;

    // dataset property
    int         N;
    int         D;
    METRIC      metric;

    // juno impl property
    bool        use_pq;
    int         coarse_grained_cluster_num;
    RT_MODE     rt_mode;   

    // data
    T**         search_points;
    T**         cluster_centroids;
    T*          cluster_centroids_flatten;
    T*          square_C;               // [ Offline]: coarse_grained_cluster_num * D
    T*          square_Q;               // [  Online]: query_batch_size * D
    int*        search_points_labels;
    T           radius;
    T**         stat;

    // BVH dict
    std::map<int, juno_rt<T>*> bvh_dict;
    CUstream    stream;
     
public:
    juno_core(std::string _dataset_dir, 
              DATASET ds=CUSTOM, 
              T _radius=1.0,
              bool _use_pq=false, 
              int _coarse_grained_cluster_num=100, 
              RT_MODE _rt_mode=QUERY_AS_RAY
             ) 
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
        dataset_dir = _dataset_dir;
        radius = _radius;
        switch (ds) {
            case SIFT1M:
                N = 1000000;
                D = 128;
                metric = METRIC_L2;
                break;
            case SIFT1B:

                break;
            case TTI1M:

                break;
            case TTI1B:

                break;
            case CUSTOM:
            default:

                break;
        }
        use_pq = _use_pq;
        coarse_grained_cluster_num = _coarse_grained_cluster_num;
        rt_mode = _rt_mode;
        
        dbg("Begin Reading Dataset and Cluster Info.");
        search_points = new T* [N];
        for (int i = 0; i < N; i++) search_points[i] = new T[D];
        read_search_points<T>((dataset_dir + "search_points").c_str(), search_points, N, D);

        cluster_centroids = new T* [coarse_grained_cluster_num];
        cluster_centroids_flatten = new T[coarse_grained_cluster_num * D];
        for (int i = 0; i < coarse_grained_cluster_num; i++) cluster_centroids[i] = new T[D];
        read_cluster_centroids<T>((dataset_dir + "cluster_centroids").c_str(), cluster_centroids, coarse_grained_cluster_num, D);
        square_C = new T[coarse_grained_cluster_num];
        for (int i = 0; i < coarse_grained_cluster_num; i++) {
            T res = 0.0;
            for (int j = 0; j < D; j++) {
                res += cluster_centroids[i][j] * cluster_centroids[i][j];
                cluster_centroids_flatten[i * D + j] = cluster_centroids[i][j];
            }
            square_C[i] = res;
        }
        search_points_labels = new int[N];
        read_search_points_labels((dataset_dir + "search_points_labels").c_str(), search_points_labels, N);

        stat = new T*[D];
        for (int i = 0; i < D; i++) {
            stat[i] = new T[4];    // Min, Max, Mean, Std
            std::vector <T> tmp;
            tmp.clear();
            for (int j = 0; j < N; j++) {
                tmp.push_back(search_points[j][i]);
            }
            stat[i][0] = *std::min_element(tmp.begin(), tmp.end());
            stat[i][1] = *std::max_element(tmp.begin(), tmp.end());
            stat[i][2] = std::accumulate(tmp.begin(), tmp.end(), 0.0) / (1.0 * N);
            stat[i][3] = std::sqrt(std::inner_product(tmp.begin(), tmp.end(), tmp.begin(), 0.0) / (1.0 * N) - stat[i][2] * stat[i][2]);
        }
        if (use_pq == true) {
            // TODO: Load codebook and mapping(pts, codebook_entries).
        }
        dbg("Finish Reading Dataset and Cluster Info.");
    }

    void setupBVHDict() {
        OPTIX_CHECK(optixInit());
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            bvh_dict[c] = new juno_rt<T>();
            bvh_dict[c]->constructBVHforLabelWithRadius(c, search_points, search_points_labels, N, D, stat, radius, metric);
            // break;
        }
    }

    void serveQueryBatch(juno_query_batch<T>* _query_batch) {
        // square_Q[0 : Batch - 1], square_C[0 : cluster_num - 1]
        // square_Q[i] = sum([x^2 for x in     query[i]])   [ Online]
        // square_C[i] = sum([x^2 for x in centroids[i]])   [Offline]
        // QC = matmul(Queries (Batch * Dim), Centroids^T (Dim * cluster_num)) [ Online] [cuBLAS]
        // Dist[i][j] = sqrt(square_Q[i] + square_C[j] - 2 * QC[i][j])

        struct timeval st, ed;
        gettimeofday(&st, NULL);
        T** tmp = _query_batch->getQueryData();
        T* tmp_flatten = _query_batch->getFlattenQueryData();
        int query_size = _query_batch->getQuerySize();
        square_Q = new T[query_size];
        T** QC;
        T* QC_flatten;
        QC = new T*[query_size];
        int *selected_centroids = new int[query_size];
        QC_flatten = new T[query_size * coarse_grained_cluster_num];
        for (int i = 0; i < query_size; i++) QC[i] = new T[coarse_grained_cluster_num];
        for (int q = 0; q < query_size; q++) {
            T res = 0.0;
            for (int d = 0; d < D; d++) {
                res += tmp[q][d] * tmp[q][d];
            }
            square_Q[q] = res;
        }
        
        // for (int i = 0; i < query_size; i++) {
        //     for (int j = 0; j < coarse_grained_cluster_num; j++) {
        //         QC[i][j] = 0.0f;
        //         for (int k = 0; k < D; k++) {
        //             QC[i][j] += tmp[i][k] * cluster_centroids[j][k];
        //         }
        //     }
        // }
#if COARSE_GRAIN_CLUSTER_USE_GPU == 1
        // Don't
        // Disaster
#else
        
        if (typeid(T) == typeid(float)) {
            // export OPENBLAS_NUM_THREADS=16, ~0.35ms
            cblas_sgemm(CblasRowMajor, 
                        CblasNoTrans, 
                        CblasTrans,
                        query_size,
                        coarse_grained_cluster_num,
                        D,
                        1.0f,
                        tmp_flatten, 
                        D, 
                        cluster_centroids_flatten,
                        D, 
                        0.0f,
                        QC_flatten, 
                        coarse_grained_cluster_num);
        }
        else if (typeid(T) == typeid(double)) {
            // cblas_dgemm(CblasRowMajor, 
            //             CblasNoTrans, 
            //             CblasTrans,
            //             query_size,
            //             coarse_grained_cluster_num,
            //             D,
            //             1.0f,
            //             tmp_flatten, 
            //             D, 
            //             cluster_centroids_flatten,
            //             D, 
            //             0.0f,
            //             QC_flatten, 
            //             coarse_grained_cluster_num);
        }
#endif
        for (int i = 0; i < query_size; i++) {
            int min = 1e10, id = -1;
            for (int j = 0; j < coarse_grained_cluster_num; j++) {
                QC[i][j] = sqrt(square_C[j] + square_Q[i] - 2 * QC_flatten[i * coarse_grained_cluster_num + j]);
                if (QC[i][j] < min) {
                    min = QC[i][j];
                    id = j;
                }
            }
            selected_centroids[i] = id;
        }
        gettimeofday(&ed, NULL);
        elapsed("Coarse Grained Clustering", st, ed);

        gettimeofday(&st, NULL);

        float3* ray_origins = new float3[query_size * D / 2];
        for (int q = 0; q < query_size; q++) {
            for (int d = 0; d < D; d+=2) {
                float x = 1.0 * tmp[q][d], y = 1.0 * tmp[q][d + 1];
                ray_origins[q * (D >> 1) + (d >> 1)] = make_float3(x, y, 1.0 * (d >> 1) + 0.5);
            }
        }
        for (int i = 0; i < query_size; i++) {
            bvh_dict[selected_centroids[i]]->setRayOrigin(ray_origins + i * 64, 64);
        }
        for (int i = 0; i < query_size; i++) {
            auto pipeline = bvh_dict[selected_centroids[i]]->getOptixPipeline();
            auto d_param = bvh_dict[selected_centroids[i]]->getDparams();
            auto sbt = bvh_dict[selected_centroids[i]]->getSBT();
            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), sbt, 64, 1, 1));
            // @TODO: 1. can't get correct hit report with correct primitive number
            //        2. Though we can, the time will be 2ms for query=100, QPS is 50000, bad.
        }
        CUDA_SYNC_CHECK();
        dbg(query_size);
        gettimeofday(&ed, NULL);
        elapsed("Ray Tracing Intersection Test", st, ed);
    }
}; // class juno_core

}; // namespace juno

#endif