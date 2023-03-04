#include "cuda.h"
#include "cuda_fp16.h"
#include "juno_gpu_kernel.cuh"
#include <stdio.h>

// cuCollections hashmap
// #include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

// #define DEBUG_GPU 1

const float L2max = 1964.62;
namespace juno {

__global__ void calcDensity(float* p, float* q, int* res, int _N, int _Q, int _D, float r) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt = 0;
    if (t < _Q) {
        for (int i = 0; i < _N; i++) {
            float l2 = 0.0;
            for (int d = 0; d < _D; d++) {
                l2 += (p[i * _D + d] - q[t * _D + d]) * (p[i * _D + d] - q[t * _D + d]);
            }
            if (l2 < r * r) cnt++;
        }
        res[t] = cnt;
    }
}

__global__ void calcTop1000R(float* p, float* q, float* res, int _N, int _D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < _N) {
        float l2 = 0.0f;
        for (int d = 0; d < _D; d++) {
            l2 += (p[t * _D + d] - q[d]) * (p[t * _D + d] - q[d]);
        }
        res[t] = sqrt(l2);
    }
}

__global__ void calcTop100RPerDim(float* p, float *q, int* gt, float* res, int _N, int _Q, int _D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < _Q) {
        for (int d = 0; d < _D / 2; d++) {
            float r = 0.0f, x = q[t * _D + (d << 1)], y = q[t * _D + (d << 1) + 1];
            for (int g = 0; g < 100; g++) {
                int idx = gt[g];
                float _x = p[idx * _D + (d << 1)], _y = p[idx * _D + (d << 1) + 1];
                float _r = sqrt((x - _x) * (x - _x) + (y - _y) * (y - _y));
                if (_r > r) r = _r;
            }
            res[t * _D / 2 + d] = r;
        }
    }
}

__global__ void calcDensitySearchPoint(float* p, float* q, float* dmax, float* res, int _N, int _Q, int _D) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t < _Q) {
        int cnt = 0;
        for (int n = 0; n < _N; n++) {
            float l2dist = 0.0f;
            for (int d = 0; d < _D; d++) {
                l2dist += ((q[t * _D + d]) - (p[n * _D + d])) * ((q[t * _D + d]) - (p[n * _D + d]));
            }
            l2dist = sqrt(l2dist);
            if (l2dist < 0.2 * L2max) {
                cnt ++;
            }
        }
        res[t * _D] = 1.0 * cnt;
    }
    // if (t < _Q) {
    //     for (int d = 0; d < _D / 2; d++) {
    //         float r = 0.005 * sqrt(dmax[2 * d] * dmax[2 * d] + dmax[2 * d + 1] * dmax[2 * d + 1]);
    //         int cnt = 0;
    //         int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0, cnt5 = 0;
    //         float x1, x2, y1, y2, dist, mean, std, diff = 0.0;
    //         for (int n = 0; n < _N; n++) {
    //             x1 = q[t * _D + 2 * d], x2 = p[n * _D + 2 * d], y1 = q[t * _D + 2 * d + 1], y2 = p[n * _D + 2 * d + 1];
    //             dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    //             if (dist < r) {
    //                 cnt++;
    //                 if ((x1 < x2) && (y1 <= y2)) cnt1++;
    //                 else if ((x1 >= x2) & (y1 < y2)) cnt2++;
    //                 else if ((x1 > x2) & (y1 >= y2)) cnt3++;
    //                 else if ((x1 <= x2) & (y1 > y2)) cnt4++;
    //                 else cnt5++;
    //             }
    //             diff = sqrt(1.0 * ((1.0 * cnt3 - 1.0 * cnt1) * (1.0 * cnt3 - 1.0 * cnt1) + (1.0 * cnt2 - 1.0 * cnt4) * (1.0 * cnt2 - 1.0 * cnt4)));
    //         }
    //         res[t * _D + d] = cnt;
    //         res[t * _D + _D / 2 + d] = diff;
    //     }
    // }
}

void testHashmap () {
    using Key   = int;
    using Value = int8_t;

    Key constexpr empty_key_sentinel     = -1;
    Value constexpr empty_value_sentinel = -1;

    std::size_t const num_keys = 10000 ;
    std::size_t const capacity = 20000;

    thrust::device_vector<Key> insert_keys(num_keys);
    thrust::sequence(insert_keys.begin(), insert_keys.end(), 0);
    thrust::device_vector<Value> insert_values(num_keys);
    thrust::sequence(insert_values.begin(), insert_values.end(), 0);

    // cuco::static_map<Key, Value> map{
    // capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};
    // auto device_insert_view = map.get_device_mutable_view();
}

// query_cluster_mapping_array: [nq, nlists, 2]
// cluster_bias: [coarse_grained_cluster_num]
// cluster_query_mapping: [coarse_grained_cluster_num]
// d_inversed_codebook_map: [coarse_grained_cluster_num, D / M, 32]
__global__ void query_counter (int *query_cluster_mapping_array, int *cluster_bias, 
    int *cluster_query_mapping, int *cluster_query_mapping_size, 
    int *inversed_codebook_map, int *inversed_codebook_map_size, int *inversed_codebook_map_start_address, 
    unsigned int *hit_record, 
    int query_size, int nlists, int D, int M, 
    __half *counter
) {
    // int q = blockIdx.x * blockDim.x + threadIdx.x;
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int q = threadID / ((D / M) * nlists * 32), d = threadID % ((D / M) * nlists * 32) / (nlists * 32), nl = threadID % (nlists * 32) / 32, bit = threadID % 32;
    // int q = threadID / nlists, nl = threadID % nlists;
    // if (q < query_size && nl < nlists) {
    if (q < query_size && d < (D / M) && nl < nlists && bit < 32) {
        // for (int nl = 0; nl < nlists; nl ++) {
        int tmp_cluster = query_cluster_mapping_array[q * nlists * 2 + nl * 2];
        int query_in_cluster_id = query_cluster_mapping_array[q * nlists * 2 + nl * 2 + 1];
#if VERBOSE == 1
        printf("Query: %d, Cluster: %d, Bias: %d\n", q, tmp_cluster, query_in_cluster_id);
#endif
        int base_addr = cluster_bias[tmp_cluster] * D / M;
        int stride = cluster_query_mapping_size[tmp_cluster] ;

        unsigned int one = 1, zero = 0;
        // for (int d = 0; d < D / M; d++) {
            unsigned int hit_res = hit_record[base_addr + query_in_cluster_id + d * stride];
            // for (unsigned int bit = 0; bit < 32; bit++) {
                if ((hit_res & (one << bit)) != zero) {
                    int begin_addr = inversed_codebook_map_start_address[tmp_cluster * (D / M) * 32 + d * 32 + bit] ;
                    int cur_size = inversed_codebook_map_size[tmp_cluster * (D / M) * 32 + d * 32 + bit] ;
                    for (int i = 0; i < cur_size; i ++) {
                        // __hadd (counter[q * 1000000 + inversed_codebook_map[begin_addr + i]], __half (1)) ;
                        atomicAdd (&counter[q * 1000000 + inversed_codebook_map[begin_addr + i]], __half (1)) ;
                    }
                }
            // }
        // }
    // }
    }
}

void counterOnGPU (std::vector<std::vector<std::pair<int, int>>> &query_cluster_mapping, int query_size, int nlists, int D, int M, int coarse_grained_cluster_num, 
    int cluster_bias[], std::vector<std::vector<int>> &cluster_query_mapping, 
    unsigned int *d_hit_record, int index_bias, 
    std::vector<int>*** inversed_codebook_map
) {
    // init part
    int *inversed_codebook_map_size = new int [coarse_grained_cluster_num * (D / M) * 32] ;
    int inversed_codebook_map_total_size = 0 ;
    for (int c = 0; c < coarse_grained_cluster_num; c ++)
        for (int d = 0; d < D / M; d ++)
            for (int e = 0; e < 32; e ++) {
                inversed_codebook_map_size[c * (D / M) * 32 + d * 32 + e] = inversed_codebook_map[c][d][e].size() ;
                inversed_codebook_map_total_size += inversed_codebook_map[c][d][e].size() ;
            }
    int *inversed_codebook_map_array = new int [inversed_codebook_map_total_size] ;
    int *inversed_codebook_map_start_address = new int [coarse_grained_cluster_num * (D / M) * 32] ;
    int cur_pos = 0 ;
    for (int c = 0; c < coarse_grained_cluster_num; c ++)
        for (int d = 0; d < D / M; d ++)
            for (int e = 0; e < 32; e ++) {
                inversed_codebook_map_start_address[c * (D / M) * 32 + d * 32 + e] = cur_pos ;
                for (int i = 0; i < inversed_codebook_map[c][d][e].size(); i ++) {
                    inversed_codebook_map_array[cur_pos + i] = inversed_codebook_map[c][d][e][i] ;
                }
                cur_pos += inversed_codebook_map[c][d][e].size() ;
            }

    int* d_inversed_codebook_map_array, *d_inversed_codebook_map_array_size, *d_inversed_codebook_map_start_address ;
    CUDA_CHECK(cudaMalloc((void**)&d_inversed_codebook_map_array, sizeof(int) * inversed_codebook_map_total_size)) ;
    CUDA_CHECK(cudaMalloc((void**)&d_inversed_codebook_map_array_size, sizeof(int) * coarse_grained_cluster_num * (D / M) * 32)) ;
    CUDA_CHECK(cudaMalloc((void**)&d_inversed_codebook_map_start_address, sizeof(int) * coarse_grained_cluster_num * (D / M) * 32)) ;
    CUDA_CHECK(cudaMemcpy(d_inversed_codebook_map_array, inversed_codebook_map_array, sizeof(int) * inversed_codebook_map_total_size, cudaMemcpyHostToDevice)) ;
    CUDA_CHECK(cudaMemcpy(d_inversed_codebook_map_array_size, inversed_codebook_map_size, sizeof(int) * coarse_grained_cluster_num * (D / M) * 32, cudaMemcpyHostToDevice)) ;
    CUDA_CHECK(cudaMemcpy(d_inversed_codebook_map_start_address, inversed_codebook_map_start_address, sizeof(int) * coarse_grained_cluster_num * (D / M) * 32, cudaMemcpyHostToDevice)) ;

    int* d_cluster_bias ;
    CUDA_CHECK(cudaMalloc((void**)&d_cluster_bias, sizeof(int) * 1000));
    CUDA_CHECK(cudaMemcpy((void*)d_cluster_bias, (void*)cluster_bias, sizeof(int) * 1000, cudaMemcpyHostToDevice));

    __half *d_counter ;
    CUDA_CHECK(cudaMalloc((void**)&d_counter, sizeof(__half) * query_size * 1000000)) ;

    // query related
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int query_cluster_mapping_array[query_size][nlists][2];
    for (int q = 0; q < query_size; q ++)
        for (int nl = 0; nl < nlists; nl ++) {
            query_cluster_mapping_array[q][nl][0] = query_cluster_mapping[q][nl].first;
            query_cluster_mapping_array[q][nl][1] = query_cluster_mapping[q][nl].second;
        }

    int *cluster_query_mapping_array_size = new int [cluster_query_mapping.size()] ;
    int cluster_query_mapping_array_total_size = 0 ;
    for (int i = 0; i < cluster_query_mapping.size(); i ++) {
        cluster_query_mapping_array_size[i] = cluster_query_mapping[i].size() ;
        cluster_query_mapping_array_total_size += cluster_query_mapping[i].size() ;
    }
    int *cluster_query_mapping_array = new int [cluster_query_mapping_array_total_size] ;
    cur_pos = 0 ;
    for (int i = 0; i < cluster_query_mapping.size(); i ++) {
        for (int j = 0; j < cluster_query_mapping[i].size(); j ++) {
            cluster_query_mapping_array[cur_pos + j] = cluster_query_mapping[i][j] ;
        }
        cur_pos += cluster_query_mapping[i].size() ;
    }

    int* d_cluster_query_mapping_array, *d_cluster_query_mapping_array_size ;
    CUDA_CHECK(cudaMalloc((void**)&d_cluster_query_mapping_array, sizeof(int) * cluster_query_mapping_array_total_size)) ;
    CUDA_CHECK(cudaMalloc((void**)&d_cluster_query_mapping_array_size, sizeof(int) * cluster_query_mapping.size())) ;
    CUDA_CHECK(cudaMemcpy(d_cluster_query_mapping_array, cluster_query_mapping_array, sizeof(int) * cluster_query_mapping_array_total_size, cudaMemcpyHostToDevice)) ;
    CUDA_CHECK(cudaMemcpy(d_cluster_query_mapping_array_size, cluster_query_mapping_array_size, sizeof(int) * cluster_query_mapping.size(), cudaMemcpyHostToDevice)) ;

    int* d_query_cluster_mapping_array;
    CUDA_CHECK(cudaMalloc((void**)&d_query_cluster_mapping_array, sizeof(int) * query_size * nlists * 2));
    CUDA_CHECK(cudaMemcpy((void*)d_query_cluster_mapping_array, (void*)query_cluster_mapping_array, sizeof(int) * query_size * nlists * 2, cudaMemcpyHostToDevice));

    int thread_cnt = query_size * (D / M) * nlists * 32 ;
    query_counter<<<(thread_cnt + 1023) / 1024, 1024>>> (d_query_cluster_mapping_array, d_cluster_bias, d_cluster_query_mapping_array, d_cluster_query_mapping_array_size, d_inversed_codebook_map_array, d_inversed_codebook_map_array_size, d_inversed_codebook_map_start_address, d_hit_record, query_size, nlists, D, M, d_counter) ;

    // __half *counter = new __half [query_size * 1000000] ;
    // CUDA_CHECK(cudaMemcpy(counter, d_counter, sizeof(__half) * query_size * 1000000, cudaMemcpyDeviceToHost)) ;

    printf ("%s\n", cudaGetErrorString(cudaGetLastError())) ;

#if DEBUG_GPU == 1
    for (int query = 0; query < query_size; query ++) {
        int max_cnt = 0, p = 0 ;
        for (int point = 0; point < 1000000; point ++) {
            if (counter[query * 1000000 + point] > max_cnt)
                max_cnt = counter[query * 1000000 + point], p = point ;
        }
        printf ("%d %d\n", p, max_cnt) ;
    }
#endif

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[%32s]: %010.6fms\n", "counting on GPU", milliseconds);
}

void plotQueryWithDensity(float* _search_points, float* _query, float* _centroids, int* _labels, int* _ground_truth, int _N, int _Q, int _D, int _C) {
    float* d_search_points;
    float* d_query;
    float* d_centroids;
    int* d_labels;
    int* d_ground_truth;
    int* d_res, *h_res;
    float* h_recall, *d_recall;
    float *d_l2, *h_l2;
    float *d_r, *h_r;
    float dim_max[128] = {172.0, 148.0, 162.0, 158.0, 162.0, 169.0, 188.0, 158.0, 207.0, 188.0, 176.0, 158.0, 162.0, 166.0, 188.0, 193.0, 218.0, 167.0, 176.0, 169.0, 171.0, 162.0, 166.0, 198.0, 185.0, 159.0, 176.0, 160.0, 163.0, 162.0, 161.0, 164.0, 182.0, 163.0, 160.0, 161.0, 167.0, 189.0, 180.0, 169.0, 207.0, 167.0, 178.0, 158.0, 165.0, 163.0, 180.0, 173.0, 218.0, 173.0, 178.0, 163.0, 172.0, 163.0, 173.0, 193.0, 185.0, 161.0, 178.0, 169.0, 172.0, 174.0, 166.0, 159.0, 178.0, 159.0, 185.0, 162.0, 166.0, 167.0, 169.0, 168.0, 207.0, 168.0, 185.0, 163.0, 166.0, 159.0, 171.0, 176.0, 218.0, 168.0, 185.0, 161.0, 171.0, 167.0, 171.0, 188.0, 182.0, 156.0, 170.0, 164.0, 172.0, 179.0, 179.0, 166.0, 178.0, 157.0, 176.0, 174.0, 166.0, 153.0, 155.0, 164.0, 207.0, 184.0, 176.0, 157.0, 165.0, 159.0, 170.0, 204.0, 218.0, 184.0, 176.0, 159.0, 170.0, 187.0, 181.0, 191.0, 187.0, 152.0, 161.0, 150.0, 165.0, 171.0, 173.0, 161.0};
    float* d_dim_max;
    float* h_res128, *d_res128;
    h_res = new int[_Q];
    h_l2 = new float[_N];
    h_r = new float[_Q * _D / 2];
    h_res128 = new float[_Q * _D];
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_search_points), sizeof(float) * _N * _D));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query), sizeof(float) * _Q * _D));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centroids), sizeof(float) * _C * _D));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_labels), sizeof(int) * _N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ground_truth), sizeof(int) * _Q * 100));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_res), sizeof(int) * _Q));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l2), sizeof(float) * _N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_r), sizeof(float) * _Q * _D / 2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_recall), sizeof(float) * _Q));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dim_max), sizeof(float) * _D));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_res128), sizeof(float) * _Q * _D));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_search_points), _search_points, sizeof(float) * _N * _D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_query), _query, sizeof(float) * _Q * _D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ground_truth), _ground_truth, sizeof(int) * _Q * 100, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centroids), _centroids, sizeof(float) * _C * _D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_labels), _labels, sizeof(int) * _N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_dim_max), dim_max, sizeof(float) * _D, cudaMemcpyHostToDevice));

    calcDensitySearchPoint<<<40, 256>>>(d_search_points, d_query, d_dim_max, d_res128, _N, _Q, _D);
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaMemcpy(h_res128, reinterpret_cast<void*>(d_res128), sizeof(float) * _Q * _D, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 1; i++) {
    //     for (int d = 0; d < _D / 2; d++) {
    //         double r = 0.005 * sqrt(dim_max[d * 2] * dim_max[d * 2] + dim_max[d * 2 + 1] * dim_max[d * 2 + 1]);
    //         int cnt = 0;
    //         int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0, cnt5 = 0, diff = 0;
    //         double x1, x2, y1, y2, dist;
    //         for (int n = 0; n < _N; n++) {
    //             x1 = _query[i * _D + d * 2];
    //             x2 = _search_points[n * _D + d * 2];
    //             y1 = _query[i * _D + d * 2 + 1];
    //             y2 = _search_points[n * _D + d * 2 + 1];
    //             dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    //             if (dist < r) {
    //                 cnt++;

    //             x1 = _query[i * _D + d * 2];
    //             x2 = _search_points[n * _D + d * 2];
    //             y1 = _query[i * _D + d * 2 + 1];
    //             y2 = _search_points[n * _D + d * 2 + 1];
    //                 if ((x1 < x2) && (y1 <= y2)) cnt1++;
    //                 else if ((x1 >= x2) & (y1 < y2)) cnt2++;
    //                 else if ((x1 > x2) & (y1 >= y2)) cnt3++;
    //                 else if ((x1 <= x2) & (y1 > y2)) cnt4++;
    //                 // else {
    //                 //     printf("Fuck:%f, %f, %f, %f\n", x1, x2, y1, y2);
    //                 // }
    //             }

    //         }   
    //         h_res128[i * _D + d] = cnt;   
    //         h_res128[i * _D + _D / 2 + d] = cnt1 + cnt2 + cnt3 + cnt4;         
    //     }
    // }
    
    for (int i = 0; i < _Q; i++) {
        for (int d = 0; d < _D; d++) {
            printf("%5.1f, ", _query[i * _D + d]);
        }
        for (int d = 0; d < 1; d++) {
            printf("%16.6f%c", h_res128[i * _D + d], '\n');
        }
    }
    // // float radius = 500.0;
    // // calcDensity<<<40, 256>>>(d_search_points, d_query, d_res, _N, _Q, _D, radius);
    // CUDA_SYNC_CHECK();
    // CUDA_CHECK(cudaMemcpy(h_res, reinterpret_cast<void*>(d_res), sizeof(int) * _Q, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < _Q; i++) {
    //     printf("%d, ", h_res[i]);
    // }

    // for (int q = 0; q < _Q; q++) {
    //     calcTop1000R<<<1000, 1024>>>(d_search_points, d_query + q * _D, d_l2, _N, _D);
    //     CUDA_SYNC_CHECK();
    //     CUDA_CHECK(cudaMemcpy(h_l2, reinterpret_cast<void*>(d_l2), sizeof(float) * _N, cudaMemcpyDeviceToHost));
    //     std::sort(h_l2, h_l2 + _N);
    //     float query_l2 = 0.0f;
    //     for (int d = 0; d < _D; d++) {
    //         printf("%5.1f, ", _query[q * _D + d]);
    //     }
    //     printf("%11.6f, %11.6f\n", h_l2[999], h_l2[99]);
    // }
    // calcTop100RPerDim<<<40, 256>>>(d_search_points, d_query, d_ground_truth, d_r, _N, _Q, _D);
    // CUDA_SYNC_CHECK();
    // CUDA_CHECK(cudaMemcpy(h_r, reinterpret_cast<void*>(d_r), sizeof(float) * _Q * _D / 2, cudaMemcpyDeviceToHost));
    // for (int q = 0; q < _Q; q++) {
    //     for (int d = 0; d < _D; d++) {
    //         printf("%5.1f, ", _query[q * _D + d]);
    //     }
    //     for (int d = 0; d < _D / 2; d++) {
    //         printf("%11.6f%c", h_r[q * _D / 2 + d], (d == _D / 2 - 1) ? '\n' : ',');
    //     }
    // }
}

float L2dist(float* a, float* b, int len) {
    float res = 0.0f;
    for (int i = 0; i < len; i++) {
        res += (a[i] - b[i]) * (a[i] - b[i]);
    }
    res = sqrt(res);
    return res;
}

void referenceModel(float* _search_points, float* _query, float* _centroids, int* _labels, int* _ground_truth, int _N, int _Q, int _D, int _C, float** stat) {
    float recalls = 0.0;
    for (int q = 0; q < 1; q++) {
        int selected_cluster = -1;
        float L2 = 1e10;
        for (int c = 0; c < _C; c++) {
            float tmp = L2dist(_query + q * _D, _centroids + c * _D, _D);
            if (tmp < L2) {
                L2 = tmp;
                selected_cluster = c;
            }
        }
        std::vector <int> candidate;
        for (int n = 0; n < _N; n++) {
            if (_labels[n] == selected_cluster) {
                candidate.push_back(n);
            }
        }
        int P = candidate.size();
        float factors[64] = {226.91, 226.292, 234.105, 245.577, 279.63, 236.516, 231.948, 269.431, 274.614, 244.002, 235.553, 258.38, 243.939, 237.857, 229.811, 229.819, 244.322, 226.982, 252.21, 246.903, 265.966, 238.008, 231.935, 249.658, 278.304, 241.357, 236.966, 259.187, 245.247, 245.449, 244.663, 229.863, 238.673, 245.904, 235.468, 238.296, 266.595, 246.564, 229.863, 245.392, 275.224, 245.247, 239.019, 254.136, 239.708, 236.212, 248.244, 244.125, 237.346, 247.491, 225.754, 225.657, 276.957, 235.85, 229.142, 265.548, 285.272, 237.186, 252.723, 263.139, 240.983, 220.048, 237.626, 236.326};
        std::vector <std::pair<int, int>> candidate_with_cnt;
        for (int p = 0; p < 16; p++) {
            int cnt = 0;
            for (int d = 0; d < _D; d+=2) {
                float x1 = _search_points[candidate[p] * _D + d], y1 = _search_points[candidate[p] * _D + d + 1];
                float x2 = _query[q * _D + d], y2 = _query[q * _D + d + 1];
                // if (sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) < 0.25 * factors[d / 2]) {
                //     cnt ++;
                // }
                // if (max(abs(x2 - x1), abs(y2 - y1)) < 0.25 * factors[d / 2]) {
                //     cnt ++;
                // }
                // std::cout << x1 << ", " << y1 << " || " << x2 << ", " << y2 << " || Dist: " << max(abs(x2 - x1), abs(y2 - y1)) << " " << ((max(abs(x2 - x1), abs(y2 - y1)) < 0.25 * factors[d / 2]) ? "Hit" : "Miss") << std::endl;
                std::cout << ((max(abs(x2 - x1), abs(y2 - y1)) < 0.25 * factors[d / 2]) ? "Hit" : "Miss") << std::endl;
            }
            candidate_with_cnt.push_back(std::pair<int, int>(candidate[p], cnt));
        }
        for (auto && cwc : candidate_with_cnt) {
            std::cout << cwc.first << " " << cwc.second << std::endl;
        }
        // sort(candidate_with_cnt.begin(), candidate_with_cnt.end(), [](std::pair<int, int> a, std::pair<int, int> b) { return a.second > b.second; });
        // int recall = 0;
        // for (int r = 0; r < 1000; r++) {
        //     for (int gt = 0; gt < 100; gt++) {
        //         if (candidate_with_cnt[r].first == _ground_truth[q * 100 + gt]) {
        //             recall++;
        //             break;
        //         }
        //     }
        // }
        // recalls += 1.0 * recall;
    }
    // std::cout << recalls / (1.0 * _Q) << std::endl;
}

}; // namespace juno