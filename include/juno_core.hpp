#pragma once
#ifndef JUNO_CORE_H_
#define JUNO_CORE_H_

#include "utils.hpp"
#include "juno_rt.hpp"
#include "juno_query.hpp"
namespace juno {

template <typename T>
class juno_core {
private:
    std::string dataset_dir;

    // dataset property
    int         N;
    int         M;
    int         D;
    int         Q;
    int         PQ_entry;
    METRIC      metric;

    // juno impl property
    bool        use_pq;
    int         coarse_grained_cluster_num;
    RT_MODE     rt_mode;   

    // data
    T**         search_points;
    T*          search_points_flatten;
    T**         cluster_centroids;
    T*          cluster_centroids_flatten;
    std::vector<std::pair<int, std::vector <T>>> cluster_centroids_vec;
    T*          square_C;               // [ Offline]: coarse_grained_cluster_num * D
    T*          square_Q;               // [  Online]: query_batch_size * D
    int*        search_points_labels;
    int**       ground_truth;
    int*        ground_truth_flatten;
    T           radius;
    T**         stat;
    T****       codebook_entry;         // [C][D/M][E][M]
    int***      codebook_labels;        // [C][D/M][]
    std::vector<int>*** inversed_codebook_map; // [C][D/M][32][]
    
    std::map<int, std::vector<int>> points_cluster_mapping;

    // BVH dict
    std::map<int, juno_rt<T>*> bvh_dict;
    CUstream    stream;
    float factors[64] = {226.91, 226.292, 234.105, 245.577, 279.63, 236.516, 231.948, 269.431, 274.614, 244.002, 235.553, 258.38, 243.939, 237.857, 229.811, 229.819, 244.322, 226.982, 252.21, 246.903, 265.966, 238.008, 231.935, 249.658, 278.304, 241.357, 236.966, 259.187, 245.247, 245.449, 244.663, 229.863, 238.673, 245.904, 235.468, 238.296, 266.595, 246.564, 229.863, 245.392, 275.224, 245.247, 239.019, 254.136, 239.708, 236.212, 248.244, 244.125, 237.346, 247.491, 225.754, 225.657, 276.957, 235.85, 229.142, 265.548, 285.272, 237.186, 252.723, 263.139, 240.983, 220.048, 237.626, 236.326};
    
    unsigned int* hit_record;
public:
    juno_core(std::string _dataset_dir, 
              DATASET ds=CUSTOM, 
              T _radius=0.1,
              bool _use_pq=true, 
              int _coarse_grained_cluster_num=1000, 
              RT_MODE _rt_mode=QUERY_AS_RAY
             ) 
    {
        M = 2;
        CUDA_CHECK(cudaStreamCreate(&stream));
        dataset_dir = _dataset_dir;
        radius = _radius;
        switch (ds) {
            case SIFT1M:
                N = 1000000;
                D = 128;
                Q = 10000;
                PQ_entry = 32;
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
        hit_record = new unsigned int[QUERY_BATCH_MAX * NLISTS_MAX * (D / M)];
        dbg("Begin Reading Dataset and Cluster Info.");
        search_points = new T* [N];
        search_points_flatten = new T[N * D];
        for (int i = 0; i < N; i++) search_points[i] = new T[D];
        read_search_points<T>((dataset_dir + "search_points").c_str(), search_points, N, D);
        for (int n = 0; n < N; n++) {
            for (int d = 0; d < D; d++) {
                search_points_flatten[n * D + d] = search_points[n][d];
            }
        }

        cluster_centroids_vec.clear();
        cluster_centroids = new T* [coarse_grained_cluster_num];
        cluster_centroids_flatten = new T[coarse_grained_cluster_num * D];
        for (int i = 0; i < coarse_grained_cluster_num; i++) cluster_centroids[i] = new T[D];
        read_cluster_centroids<T>((dataset_dir + "cluster_centroids_" + std::to_string(coarse_grained_cluster_num)).c_str(), cluster_centroids, coarse_grained_cluster_num, D);
        square_C = new T[coarse_grained_cluster_num];
        std::vector <T> centroid;
        for (int i = 0; i < coarse_grained_cluster_num; i++) {
            T res = 0.0;
            centroid.clear();
            for (int j = 0; j < D; j++) {
                res += cluster_centroids[i][j] * cluster_centroids[i][j];
                cluster_centroids_flatten[i * D + j] = cluster_centroids[i][j];
                centroid.push_back(cluster_centroids[i][j]);
            }
            cluster_centroids_vec.push_back(std::pair<int, std::vector<T>>(i, centroid));
            square_C[i] = res;
        }
        search_points_labels = new int[N];
        read_search_points_labels((dataset_dir + "search_points_labels").c_str(), search_points_labels, N);
        for (int n = 0; n < N; n++) {
            int label = search_points_labels[n];
            points_cluster_mapping[label].push_back(n);
        }
        int* cluster_size = new int[coarse_grained_cluster_num];
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            int cnt = 0;
            for (int s = 0; s < N; s++) {
                if (search_points_labels[s] == c) {
                    cnt++;
                }
            }
            cluster_size[c] = cnt;
        }

        ground_truth = new int*[Q];
        ground_truth_flatten = new int[Q * 100];
        for (int i = 0; i < Q; i++) ground_truth[i] = new int[100];
        read_ground_truth((dataset_dir + "ground_truth").c_str(), ground_truth, Q);
        for (int q = 0; q < Q; q++) {
            for (int gt = 0; gt < 100; gt++) {
                ground_truth_flatten[q * 100 + gt] = ground_truth[q][gt];
            }
        }

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
            codebook_entry = new T***[coarse_grained_cluster_num];
            for (int c = 0; c < coarse_grained_cluster_num; c++) {
                codebook_entry[c] = new T**[D / M];
                for (int d = 0; d < D / M; d++) {
                    codebook_entry[c][d] = new T*[PQ_entry];
                    for (int e = 0; e < PQ_entry; e++) {
                        codebook_entry[c][d][e] = new T[PQ_DIM];
                    }
                }
            }

            codebook_labels = new int**[coarse_grained_cluster_num];
            for (int c = 0; c < coarse_grained_cluster_num; c++) {
                codebook_labels[c] = new int*[D / M];
                for (int d = 0; d < D / M; d++) {
                    codebook_labels[c][d] = new int[cluster_size[c]];
                }
            }
            read_codebook_entry_labels(dataset_dir + "codebook_" + std::to_string(coarse_grained_cluster_num), codebook_entry, codebook_labels, cluster_size, coarse_grained_cluster_num, PQ_entry, D);
            inversed_codebook_map = new std::vector<int>** [coarse_grained_cluster_num];
            for (int c = 0; c < coarse_grained_cluster_num; c++) {
                inversed_codebook_map[c] = new std::vector<int>* [D / M];
                for (int d = 0; d < D / M; d++) {
                    inversed_codebook_map[c][d] = new std::vector<int> [32];
                    for (int e = 0; e < 32; e++) {
                        for (int n = 0; n < cluster_size[c]; n++) {
                            if (codebook_labels[c][d][n] == e) {
                                inversed_codebook_map[c][d][e].push_back(n);
                            }
                        }
                    }
                }
            }
        }
        dbg("Finish Reading Dataset and Cluster Info.");
    }

    void setupBVHDict() {
        OPTIX_CHECK(optixInit());
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            bvh_dict[c] = new juno_rt<T>();
            bvh_dict[c]->constructBVHforLabelWithRadius(c, search_points, search_points_labels, N, D, stat, radius, metric);
            
        }
    }

    void buildJunoIndex() {
        OPTIX_CHECK(optixInit());
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            bvh_dict[c] = new juno_rt<T>();
            bvh_dict[c]->constructBVHwithPQ(c, codebook_entry, PQ_entry, D, M, stat, radius, metric);
            std::remove("/var/tmp/OptixCache_zhliu/optix7cache.db");
        }
    }

    void buildJunoIndexWhole() {
        OPTIX_CHECK(optixInit());
        bvh_dict[0] = new juno_rt<T>(Q);
        bvh_dict[0]->constructCompleteBVHwithPQ(codebook_entry, coarse_grained_cluster_num, PQ_entry, D, M, stat, radius, metric);
    }

    void serveQueryWhole(juno_query_batch<T>* _query_batch, int nlists) {
        assert((nlists < NLISTS_MAX) || "Max nlists exceeded.\n");
        // 1st filtering, can be optimized using CUDA/OpenMP
        struct timeval st, ed;
        gettimeofday(&st, NULL);
        T** query_data = _query_batch->getQueryData();
        T* query_data_flatten = _query_batch->getFlattenQueryData();
        int query_size = _query_batch->getQuerySize();
        int cluster_bias[1000] = {-1};
        std::vector<std::vector<int>> cluster_query_mapping;
        std::vector<std::vector<std::pair<int, int>>> query_cluster_mapping;
        float **L2mat = new float*[query_size];
        // Can be optimized with OpenBLAS
        for (int q = 0; q < query_size; q++) {
            L2mat[q] = new float[coarse_grained_cluster_num];
            for (int c = 0; c < coarse_grained_cluster_num; c++) {
                L2mat[q][c] = L2Dist(query_data[q], cluster_centroids[c], D);
            }
            std::vector <std::pair<int, int>> query_place_holder;
            query_cluster_mapping.push_back(query_place_holder);
        }
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            std::vector<int> query_ids;
            query_ids.clear();
            cluster_query_mapping.push_back(query_ids);
        }
        // Can be optimized use OpenMP/CUDA
        for (int q = 0; q < query_size; q++) {
            std::vector <T> query_vec;
            query_vec.clear();
            for (int d = 0; d < D; d++) {
                query_vec.push_back(query_data[q][d]);
            }
            std::sort(cluster_centroids_vec.begin(), cluster_centroids_vec.end(), [q, L2mat](const std::pair<int, std::vector <T>>& a, const std::pair<int, std::vector <T>>& b) {
                return L2mat[q][a.first] < L2mat[q][b.first];
            });
            for (int nl = 0; nl < nlists; nl++) {
                query_cluster_mapping[q].push_back(std::pair<int, int>(cluster_centroids_vec[nl].first, cluster_query_mapping[cluster_centroids_vec[nl].first].size()));
                cluster_query_mapping[cluster_centroids_vec[nl].first].push_back(q);
            }
        }
        gettimeofday(&ed, NULL);
        elapsed("Filtering", st, ed);

        gettimeofday(&st, NULL);
        // 2nd setting ray origins
        float3* ray_origin_whole = new float3[Q * (D / M) * nlists];
        int index_bias = 0, accum = 0;
        int cmax = 0;
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            cluster_bias[c] = accum;
            int query_of_cluster_c = cluster_query_mapping[c].size();
            accum += query_of_cluster_c;
            cmax = std::max(cmax, query_of_cluster_c);
            float bias = 1.0 * c;
            for (int d = 0; d < D / M; d++) {
                for (int q = 0; q < query_of_cluster_c; q++) {
                    float x = SCALE * (1.0 * query_data[cluster_query_mapping[c][q]][2 * d] / sqrt(factors[d])) + bias;
                    float y = SCALE * (1.0 * query_data[cluster_query_mapping[c][q]][2 * d + 1] / sqrt(factors[d]));
                    ray_origin_whole[index_bias++] = make_float3(x, y, 1.0 * 2 * d);
                }
            }

        }
        bvh_dict[0]->setRayOrigin(ray_origin_whole, index_bias);
        auto pipeline = bvh_dict[0]->getOptixPipeline();
        auto d_param = bvh_dict[0]->getDparams();
        auto sbt = bvh_dict[0]->getSBT();   
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), sbt, index_bias, 1, 1));
        CUDA_SYNC_CHECK();
        gettimeofday(&ed, NULL);
        elapsed("Ray Tracing", st, ed);

        gettimeofday(&st, NULL);
        bvh_dict[0]->getRayHitRecord(hit_record, index_bias);
        for (int q = 0; q < query_size; q++) {
            for (int nl = 0; nl < nlists; nl++) {
                int tmp_cluster = query_cluster_mapping[q][nl].first;
                int base_addr = cluster_bias[tmp_cluster] * D / M;
                int stride = cluster_query_mapping[tmp_cluster].size();
                std::unordered_map <int, int> point_counter_mapping;
                for (int d = 0; d < D / M; d++) {
                    for (int bit = 0; bit < 32; bit++) {
                        if (hit_record[base_addr + d * stride] & (1 << bit) == 1) {
                            for (auto && item : inversed_codebook_map[tmp_cluster][d][bit]) {
                                point_counter_mapping[item] ++;
                            }
                        }
                    }
                    // printf("%08x%c", hit_record[base_addr + d * stride], (d % 16 == 15) ? '\n' : ' ');
                }
                // printf("\n");
                // std::cout << "Cluster: " << query_cluster_mapping[q][nl].first << ", bias in cluster: " << query_cluster_mapping[q][nl].second << std::endl;
            }
        }
        gettimeofday(&ed, NULL);
        elapsed("Computing Hit Result", st, ed);

    }

    void serveQuery(juno_query_batch<T>* _query_batch, int nlists) {
        // juno_rt<T>.initRayOriginArray(Q, D, M, nlists);
        struct timeval st, ed;
        gettimeofday(&st, NULL);
        T** query_data = _query_batch->getQueryData();
        T* query_data_flatten = _query_batch->getFlattenQueryData();
        int query_size = _query_batch->getQuerySize();
        std::vector<std::vector<int>> cluster_query_mapping;
        float **L2mat = new float*[query_size];
        // Can be optimized with OpenBLAS
        for (int q = 0; q < query_size; q++) {
            L2mat[q] = new float[coarse_grained_cluster_num];
            for (int c = 0; c < coarse_grained_cluster_num; c++) {
                L2mat[q][c] = L2Dist(query_data[q], cluster_centroids[c], D);
            }
        }
        std::cout << "L2 calc complete" << std::endl;
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            std::vector<int> query_ids;
            query_ids.clear();
            cluster_query_mapping.push_back(query_ids);
        }
        // Can be optimized use OpenMP/CUDA
        for (int q = 0; q < query_size; q++) {
            std::vector <T> query_vec;
            query_vec.clear();
            for (int d = 0; d < D; d++) {
                query_vec.push_back(query_data[q][d]);
            }
            std::sort(cluster_centroids_vec.begin(), cluster_centroids_vec.end(), [q, L2mat](const std::pair<int, std::vector <T>>& a, const std::pair<int, std::vector <T>>& b) {
                return L2mat[q][a.first] < L2mat[q][b.first];
            });
            for (int nl = 0; nl < nlists; nl++) {
                cluster_query_mapping[cluster_centroids_vec[nl].first].push_back(q);
            }
        }
        gettimeofday(&ed, NULL);
        elapsed("Filtering", st, ed);
        gettimeofday(&st, NULL);
        int res = 0;
        for (int c = 0; c < 1; c++) {
            int query_of_cluster_c = cluster_query_mapping[c].size();
            float3* ray_origin = new float3[query_of_cluster_c * D / M];
            for (int d = 0; d < D / M; d++) {
                for (int q = 0; q < query_of_cluster_c; q++) {
                    float x = 1.0 * query_data[cluster_query_mapping[c][q]][2 * d];
                    float y = 1.0 * query_data[cluster_query_mapping[c][q]][2 * d + 1];
                    ray_origin[d * query_of_cluster_c + q] = make_float3(x, y, 1.0 * 2 * d);
                } 
            }
            bvh_dict[c]->setRayOrigin(ray_origin, query_of_cluster_c * D / M);
            delete [] ray_origin;
            auto pipeline = bvh_dict[c]->getOptixPipeline();
            auto d_param = bvh_dict[c]->getDparams();
            auto sbt = bvh_dict[c]->getSBT();      
            // query_of_cluster_c * D / M
            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), sbt, query_of_cluster_c * D / M, 1, 1));
            CUDA_SYNC_CHECK();
        }
        gettimeofday(&ed, NULL);
        elapsed("Ray Tracing", st, ed);
        
    }

    // Legacy
    void serveQueryBatch(juno_query_batch<T>* _query_batch) {
        // square_Q[0 : Batch - 1], square_C[0 : cluster_num - 1]
        // square_Q[i] = sum([x^2 for x in     query[i]])   [ Online]
        // square_C[i] = sum([x^2 for x in centroids[i]])   [Offline]
        // QC = matmul(Queries (Batch * Dim), Centroids^T (Dim * cluster_num)) [ Online] [cuBLAS]
        // Dist[i][j] = sqrt(square_Q[i] + square_C[j] - 2 * QC[i][j])
        unsigned int* h_hit;
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
        // elapsed("Coarse Grained Clustering", st, ed);

        float3* ray_origins = new float3[query_size * D / 2];
        for (int q = 0; q < query_size; q++) {
            for (int d = 0; d < D; d+=2) {
                float x = 1.0 * tmp[q][d], y = 1.0 * tmp[q][d + 1];
                ray_origins[q * (D >> 1) + (d >> 1)] = make_float3(x, y, 1.0 * (d >> 1) + 0.5);
            }
        }
        std::cout << selected_centroids[0] << std::endl;
        for (int i = 0; i < query_size; i++) {
            bvh_dict[selected_centroids[i]]->setRayOrigin(ray_origins + i * D / M, D / M);
        }
        std::cout << "Ray Set" << std::endl;
        gettimeofday(&st, NULL);
        for (int i = 0; i < 1; i++) {
            auto pipeline = bvh_dict[selected_centroids[i]]->getOptixPipeline();
            auto d_param = bvh_dict[selected_centroids[i]]->getDparams();
            auto sbt = bvh_dict[selected_centroids[i]]->getSBT();
            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), sbt, 64, 1, 1));
            
            CUDA_SYNC_CHECK();
            // @TODO: 1. can't get correct hit report with correct primitive number
            //        2. Though we can, the time will be 2ms for query=100, QPS is 50000, bad.
            // break;
            // CUDA_SYNC_CHECK();
            // auto device_hit = bvh_dict[selected_centroids[i]]->getPrimitiveHit();
            // int hn = bvh_dict[selected_centroids[i]]->getHitableNum();
            // CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_hit), sizeof(unsigned int) * hn));
            // CUDA_CHECK(cudaMemcpy(h_hit, reinterpret_cast<void*>(device_hit), sizeof(unsigned int) * hn, cudaMemcpyDeviceToHost));
            // std::vector <std::pair<int, int>> candidate;
            // for (int x = 0; x < hn / 64; x++) {
            //     for (int y = 0; y < 64; y++) {
            //         std::cout << h_hit[y * (hn / 64) + x] << std::endl;
            //     }
            // }
            // int g_cnt = 0;
            // for (int x = 0; x < 64; x++) {
            //     int cnt = 0;
            //     for (int y = 0; y < hn / 64; y++) {
            //         std::cout << h_hit[x * (hn / 64) + y] << std::endl;
            //     }
            //     candidate.push_back(std::pair<int, int>(points_cluster_mapping[19][x], cnt));
            // }
            // for (auto && c : candidate) {
            //     std::cout << c.first << " " << c.second << std::endl;
            // }
            // std::sort(candidate.begin(), candidate.end(), [](std::pair<int, int> a, std::pair<int, int> b) {return a.second > b.second;});
            // int recall = 0;
            // for (int top = 0; top < 100; top++) {
            //     for (int c = 0; c < 1000; c++) {
            //         if (candidate[c].first == ground_truth[0][top]) {
            //             recall++;
            //         }
            //     }
            // }

        }
        // dbg(query_size);
        gettimeofday(&ed, NULL);
        elapsed("Ray Tracing Intersection Test", st, ed);

    }

    void plotDataset(juno_query_total<T>* query_total) {
        // plotQueryWithDensity(search_points_flatten, query_total->getQueryDataFlatten(), cluster_centroids_flatten, search_points_labels, query_total->getaGroundTruthFlatten() , N, query_total->getQueryAmount(), D, coarse_grained_cluster_num);
        referenceModel(search_points_flatten, query_total->getQueryDataFlatten(), cluster_centroids_flatten, search_points_labels, query_total->getaGroundTruthFlatten() , N, query_total->getQueryAmount(), D, coarse_grained_cluster_num, stat);
    }
}; // class juno_core

}; // namespace juno

#endif