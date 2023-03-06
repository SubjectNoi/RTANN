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
    int*        cluster_size;
    T           radius;
    T**         stat;
    T****       codebook_entry;         // [C][D/M][E][M]
    int***      codebook_labels;        // [C][D/M][]
    std::vector<int>*** inversed_codebook_map; // [C][D/M][32][]
    std::vector<int>*** inversed_codebook_map_localid;
    uint8_t*    hit_res;
    int*        sub_cluster_size;       // [C * D/M * 32]
    int*        all_candidates;         // [10813582]
    int*        all_candidates_bias;    // [Q]
    int*        all_candidates_cluster; // [Q]
    unsigned int*    candidates_belong_on_every_dim; //[10813582][D / M]
    std::vector<std::vector<int>> cluster_points_mapping;
    std::vector<int>** points_belongings; // [C][D / M][]
    
    std::map<int, std::vector<int>> points_cluster_mapping;

    // BVH dict
    std::map<int, juno_rt<T>*> bvh_dict;
    CUstream    stream;
    float factors[64] = {226.91, 226.292, 234.105, 245.577, 279.63, 236.516, 231.948, 269.431, 274.614, 244.002, 235.553, 258.38, 243.939, 237.857, 229.811, 229.819, 244.322, 226.982, 252.21, 246.903, 265.966, 238.008, 231.935, 249.658, 278.304, 241.357, 236.966, 259.187, 245.247, 245.449, 244.663, 229.863, 238.673, 245.904, 235.468, 238.296, 266.595, 246.564, 229.863, 245.392, 275.224, 245.247, 239.019, 254.136, 239.708, 236.212, 248.244, 244.125, 237.346, 247.491, 225.754, 225.657, 276.957, 235.85, 229.142, 265.548, 285.272, 237.186, 252.723, 263.139, 240.983, 220.048, 237.626, 236.326};
    
    unsigned int* hit_record;
public:
    juno_core(std::string _dataset_dir, 
              DATASET ds=CUSTOM, 
              int _coarse_grained_cluster_num=1000, 
              T _radius=0.15,
              bool _use_pq=true, 
              RT_MODE _rt_mode=QUERY_AS_RAY
             ) 
    {
        // omp_set_num_threads(64);
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
            case DEEP1M:
                N = 1000000;
                D = 96;
                Q = 10000;
                PQ_entry = 32;
                metric = METRIC_L2;
                break;
            case CUSTOM:
                N = 16;
                D = 4;
                Q = 1;
                PQ_entry = 4;
                metric = METRIC_L2;
                break;
            default:

                break;
        }
        use_pq = _use_pq;
        coarse_grained_cluster_num = _coarse_grained_cluster_num;
        rt_mode = _rt_mode;
        hit_record = new unsigned int[QUERY_BATCH_MAX * NLISTS_MAX * (D / M)];
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hit_res), sizeof(uint8_t) * 10813582));
        sub_cluster_size = new int[coarse_grained_cluster_num * (D / M) * PQ_entry];
        all_candidates_bias = new int[Q];
        all_candidates_cluster = new int[Q];
        all_candidates = new int[10813582];
        candidates_belong_on_every_dim = new unsigned int[10813582 * 16];
        printf("Reading Search Points...");
        search_points = new T* [N];
        search_points_flatten = new T[N * D];
        for (int i = 0; i < N; i++) search_points[i] = new T[D];
        read_search_points<T>((dataset_dir + "search_points").c_str(), search_points, N, D);
        for (int n = 0; n < N; n++) {
            for (int d = 0; d < D; d++) {
                search_points_flatten[n * D + d] = search_points[n][d];
            }
        }
        printf("Finished\n");

        printf("Reading Cluster Centroids...");
        cluster_centroids_vec.clear();
        cluster_centroids = new T* [coarse_grained_cluster_num];
        cluster_centroids_flatten = new T[coarse_grained_cluster_num * D];
        for (int i = 0; i < coarse_grained_cluster_num; i++) cluster_centroids[i] = new T[D];
        read_cluster_centroids<T>((dataset_dir + "parameter_0/" + "cluster_centroids_" + std::to_string(coarse_grained_cluster_num)).c_str(), cluster_centroids, coarse_grained_cluster_num, D);
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
        printf("Finished\n");
        printf("Reading Search Point Labels...");
        search_points_labels = new int[N];
        read_search_points_labels((dataset_dir + "parameter_0/" + "search_points_labels_" + std::to_string(coarse_grained_cluster_num)).c_str(), search_points_labels, N);
        for (int n = 0; n < N; n++) {
            int label = search_points_labels[n];
            points_cluster_mapping[label].push_back(n);
        }
        cluster_size = new int[coarse_grained_cluster_num];
        int maxx = 0;
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            std::vector <int> place_holder;
            place_holder.clear();
            cluster_points_mapping.push_back(place_holder);
            int cnt = 0;
            for (int s = 0; s < N; s++) {
                if (search_points_labels[s] == c) {
                    cnt++;
                    cluster_points_mapping[c].push_back(s);
                }
            }
            cluster_size[c] = cnt;
        }
        printf("Finished\n");

        printf("Reading Ground Truth...");
        ground_truth = new int*[Q];
        ground_truth_flatten = new int[Q * 100];
        for (int i = 0; i < Q; i++) ground_truth[i] = new int[100];
        read_ground_truth((dataset_dir + "ground_truth").c_str(), ground_truth, Q);
        for (int q = 0; q < Q; q++) {
            for (int gt = 0; gt < 100; gt++) {
                ground_truth_flatten[q * 100 + gt] = ground_truth[q][gt];
            }
        }
        printf("Finished\n");

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
            printf("Reading Codebook Entry...");
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
            read_codebook_entry_labels(dataset_dir + "parameter_0/" + "codebook_" + std::to_string(coarse_grained_cluster_num), codebook_entry, codebook_labels, cluster_size, coarse_grained_cluster_num, PQ_entry, D);
            printf("Finished\n");
            inversed_codebook_map = new std::vector<int>** [coarse_grained_cluster_num];
            inversed_codebook_map_localid = new std::vector<int>** [coarse_grained_cluster_num];
            points_belongings = new std::vector<int>* [coarse_grained_cluster_num];
            for (int c = 0; c < coarse_grained_cluster_num; c++) {
                inversed_codebook_map[c] = new std::vector<int>* [D / M];
                inversed_codebook_map_localid[c] = new std::vector<int>* [D / M];
                points_belongings[c] = new std::vector<int> [D / M];
                for (int d = 0; d < D / M; d++) {
                    inversed_codebook_map[c][d] = new std::vector<int> [32];
                    inversed_codebook_map_localid[c][d] = new std::vector<int> [32];
                    points_belongings[c][d].clear();
                    for (int e = 0; e < PQ_entry; e++) {
                        inversed_codebook_map[c][d][e].clear();
                        inversed_codebook_map_localid[c][d][e].clear();
                        for (int n = 0; n < cluster_size[c]; n++) {
                            if (codebook_labels[c][d][n] == e) {
                                inversed_codebook_map[c][d][e].push_back(cluster_points_mapping[c][n]);
                                inversed_codebook_map_localid[c][d][e].push_back(n);
                                points_belongings[c][d].push_back(e);
                            }
                        }
                        sub_cluster_size[c * (D / M) * PQ_entry + d * PQ_entry + e] = inversed_codebook_map[c][d][e].size();
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
        std::remove("/var/tmp/OptixCache_zhliu/optix7cache.db");
        bvh_dict[0] = new juno_rt<T>(Q);
        bvh_dict[0]->constructCompleteBVHwithPQ(codebook_entry, coarse_grained_cluster_num, PQ_entry, D, M, stat, radius, metric);
    }

    void serveQueryWhole(juno_query_batch<T>* _query_batch, int nlists) {
        // omp_set_num_threads(16);
        assert((nlists < NLISTS_MAX) || "Max nlists exceeded.\n");
        // 1st filtering, can be optimized using CUDA/OpenMP
        struct timeval st, ed;
        int candidate_sum = 0;
        T** query_data = _query_batch->getQueryData();
        T* query_data_flatten = _query_batch->getFlattenQueryData();
        int query_size = _query_batch->getQuerySize();
        int cluster_bias[1000] = {-1};
        int cluster_query_size[1000] = {0};
        // Record which queries fall into the cluster C
        std::vector<std::vector<int>> cluster_query_mapping;
        int *total_candidate = new int[query_size];
        
        // Record which clusters a query falls in
        std::vector<std::vector<std::pair<int, int>>> query_cluster_mapping;
        float **L2mat = new float*[query_size];
        for (int q = 0; q < query_size; q++) L2mat[q] = new float[coarse_grained_cluster_num];
        // Can be optimized with OpenBLAS   
        gettimeofday(&st, NULL);     
        for (int q = 0; q < query_size; q++) {
            // Calculate the L2-dist between every cluster centroids
            // #pragma omp parallel for
            for (int c = 0; c < coarse_grained_cluster_num; c++) {
                L2mat[q][c] = L2Dist(query_data[q], cluster_centroids[c], D);
            }
            std::vector <std::pair<int, int>> query_place_holder;
            query_cluster_mapping.push_back(query_place_holder);
        }

        gettimeofday(&ed, NULL);
        elapsed("Calculate L2 Dist[CPU]", st, ed);

        gettimeofday(&st, NULL);
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            std::vector<int> query_ids;
            query_ids.clear();
            cluster_query_mapping.push_back(query_ids);
        }
        // Can be optimized use OpenMP/CUDA
        // #pragma omp parallel for
        for (int q = 0; q < query_size; q++) {
            all_candidates_bias[q] = candidate_sum;
            int cnt = 0;
            std::vector <T> query_vec;
            query_vec.clear();
            for (int d = 0; d < D; d++) {
                query_vec.push_back(query_data[q][d]);
            }

            // Sort by L2 distance
            std::sort(cluster_centroids_vec.begin(), cluster_centroids_vec.end(), [q, L2mat](const std::pair<int, std::vector <T>>& a, const std::pair<int, std::vector <T>>& b) {
                return L2mat[q][a.first] < L2mat[q][b.first];
            });

            // Select nlists cluster
            for (int nl = 0; nl < nlists; nl++) {
                int local_size = cluster_size[cluster_centroids_vec[nl].first];
                // Record a pair, stands for: <the cluster c this query q use, the position this query q falls in the cluster c>
                query_cluster_mapping[q].push_back(std::pair<int, int>(cluster_centroids_vec[nl].first, cluster_query_mapping[cluster_centroids_vec[nl].first].size()));
                // Push query q into the query_list of cluster c
                cluster_query_mapping[cluster_centroids_vec[nl].first].push_back(q);
                cnt += local_size;
                all_candidates_cluster[q] = cluster_centroids_vec[nl].first;
                for (int i = 0; i < local_size; i++) {
                    all_candidates[all_candidates_bias[q] + i] = q;
                    // for (int d = 0; d < D / M; d++) {
                    //     candidates_belong_on_every_dim[(all_candidates_bias[q] + i) * (D / M) + d] = points_belongings[all_candidates_cluster[q]][d][i];
                    // }
                    for (int d = 0; d < D / M; d+=4) {
                        for (int id = 0; id < 4; id++) {
                            candidates_belong_on_every_dim[(all_candidates_bias[q] + i) * 16 + d / 4] |= (unsigned int)(points_belongings[all_candidates_cluster[q]][d][i] << ((3 - id) * 8));
                        }
                    }
                }
            }
            total_candidate[q] = cnt;
            candidate_sum += cnt;
        }
        gettimeofday(&ed, NULL);
        elapsed("Filtering[CPU]", st, ed);
        // 2nd setting ray origins
        gettimeofday(&st, NULL);
        float3* ray_origin_whole = new float3[Q * (D / M) * nlists];
        int index_bias = 0, accum = 0;
        // Ray Layout: 10000 * nlists * D / 2 rays
        // [............Cluster 1 Ray.............][............Cluster 2 Ray.............]........
        // |                                       \
        // [Dim 00 Ray][Dim 01 Ray]......[Dim 63 Ray]
        // |           \                 |           \
        // [q0,q1,...,qc]                [q0,q1,...,qc]
        for (int c = 0; c < coarse_grained_cluster_num; c++) {
            int query_of_cluster_c = cluster_query_mapping[c].size();
            cluster_bias[c] = accum;
            cluster_query_size[c] = cluster_query_mapping[c].size();
            accum += query_of_cluster_c;
            float bias = 1.0 * c;
            for (int d = 0; d < D / M; d++) {
                for (int q = 0; q < query_of_cluster_c; q++) {
                    float x = (1.0 * query_data[cluster_query_mapping[c][q]][2 * d]) / 100.0;
                    float y = (1.0 * query_data[cluster_query_mapping[c][q]][2 * d + 1]) / 100.0;
                    ray_origin_whole[index_bias] = make_float3(x, y, 1.0 * (c * 128 + 2 * d));
                    index_bias++;
                }
            }
        }
        gettimeofday(&ed, NULL);
        elapsed("Setting Ray Origin[CPU]", st, ed);

        gettimeofday(&st, NULL);
        bvh_dict[0]->setRayOrigin(ray_origin_whole, index_bias);
        gettimeofday(&ed, NULL);
        elapsed("Copying Ray Origin H->D[GPU]", st, ed);
        gettimeofday(&st, NULL);
        auto pipeline = bvh_dict[0]->getOptixPipeline();
        auto d_param = bvh_dict[0]->getDparams();
        auto sbt = bvh_dict[0]->getSBT();   
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), sbt, index_bias, 1, 1));
        CUDA_SYNC_CHECK();
        gettimeofday(&ed, NULL);
        elapsed("Ray Tracing", st, ed);
        
        bvh_dict[0]->getRayHitRecord(hit_record, index_bias);
        // getHitResult(hit_record, hit_res, nlists, query_cluster_mapping, cluster_size, cluster_query_mapping, cluster_query_size, inversed_codebook_map, sub_cluster_size);
        getHitResult(hit_record, hit_res, nlists, all_candidates, all_candidates_cluster, all_candidates_bias, candidates_belong_on_every_dim);
        // uint8_t *h_hit_res;
        // h_hit_res = new uint8_t[Q * N];
        // CUDA_CHECK(cudaMemcpy(h_hit_res, reinterpret_cast<void*>(hit_res), sizeof(uint8_t) * Q * N, cudaMemcpyDeviceToHost));
        // std::cout << h_hit_res[0] << std::endl;

        /*
        gettimeofday(&st, NULL);
        bvh_dict[0]->getRayHitRecord(hit_record, index_bias);
        int r1_100 = 0;
        int r100_1000 = 0;
        // #pragma omp parallel for
        for (int q = 0; q < query_size; q++) {
            std::vector <std::pair<int, int>> sort_res;
            sort_res.clear();
            for (int nl = 0; nl < nlists; nl++) {
                int tmp_cluster = query_cluster_mapping[q][nl].first;
                int query_in_cluster_id = query_cluster_mapping[q][nl].second;
#if VERBOSE == 1
                printf("Query: %d, Cluster: %d, Bias: %d\n", q, tmp_cluster, query_in_cluster_id);
#endif
                int base_addr = cluster_bias[tmp_cluster] * D / M;
                int stride = cluster_query_mapping[tmp_cluster].size();
                std::unordered_map <int, int> point_counter_mapping;
                unsigned int one = 1, zero = 0;
                for (int d = 0; d < D / M; d++) {
                    unsigned int hit_res = hit_record[base_addr + query_in_cluster_id + d * stride];
                    for (unsigned int bit = 0; bit < 32; bit++) {
                        if ((hit_res & (one << bit)) != zero) {
                            // int cnt = 0;
                            for (auto && item : inversed_codebook_map[tmp_cluster][d][bit]) {
                                point_counter_mapping[item] ++;
                                // cnt ++;
                            }
                            // std::cout << cnt << "/" << total_candidate[q] << std::endl;
                        }
                    }
#if VERBOSE == 1
                    printf("%08x%c", hit_res, (d % 16 == 15) ? '\n' : ' ');
#endif
                }
#if VERBOSE == 1
                printf("\n");
#endif
                for (auto it = point_counter_mapping.begin(); it != point_counter_mapping.end(); it++) {
                    sort_res.push_back(std::pair<int, int>(it->first, it->second));
                }
            }
            sort(sort_res.begin(), sort_res.end(), [](const std::pair<int, int> a, const std::pair<int, int> b) {return a.second > b.second;});
            int local_r1_100 = 0;
            for (int topk = 0; topk < 100; topk++) {
                // std::cout << "(" << sort_res[topk].first << ", " << sort_res[topk].second << "), " << std::endl;
                if (sort_res[topk].first == ground_truth[q][0]) {
                    local_r1_100++;
                    break;
                }
            }
            // #pragma omp critical 
            {
                r1_100 += local_r1_100;
            }
            int local_r100_1000 = 0;
            for (int gt = 0; gt < 100; gt++) {
                for (int topk = 0; topk < 1000; topk++) {
                    if (sort_res[topk].first == ground_truth[q][gt]) {
                        local_r100_1000 ++;
                        break;
                    }
                }
            }
            // #pragma omp critical 
            // {
                r100_1000 += local_r100_1000;
            // }
        }
        
        std::cout << r1_100 << " " << (1.0 * r100_1000) / (1.0 * query_size) << std::endl;
        gettimeofday(&ed, NULL);
        elapsed("Computing Hit Result", st, ed);
        */
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
            // cblas_sgemm(CblasRowMajor, 
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