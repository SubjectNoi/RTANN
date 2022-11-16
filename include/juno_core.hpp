#pragma once
#ifndef JUNO_CORE_H_
#define JUNO_CORE_H_

#include "utils.hpp"
#include "juno_rt.hpp"

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
    int*        search_points_labels;
    T           radius;
    T**         stat;

    // BVH dict
    std::map<int, juno_rt<T>*> bvh_dict;
     
public:
    juno_core(std::string _dataset_dir, 
              DATASET ds=CUSTOM, 
              T _radius=0.3,
              bool _use_pq=false, 
              int _coarse_grained_cluster_num=100, 
              RT_MODE _rt_mode=QUERY_AS_RAY
             ) 
    {
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
        for (int i = 0; i < coarse_grained_cluster_num; i++) cluster_centroids[i] = new T[D];
        read_cluster_centroids<T>((dataset_dir + "cluster_centroids").c_str(), cluster_centroids, coarse_grained_cluster_num, D);

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
            break;
        }
    }
}; // class juno_core

}; // namespace juno

#endif