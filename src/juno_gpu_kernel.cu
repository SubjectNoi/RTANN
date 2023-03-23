#include "cuda.h"
#include "juno_gpu_kernel.cuh"
#include <stdio.h>

#include <thrust/pair.h>
#include <thrust/sort.h>

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

// query_selected_clusters: query * nlists
// points_in_codebook_entry: C * (D / M) * PQ_entry * max_numOfPointsInBit
// d_hit_record: query * nlists * 64 * 32
__global__ void gpuGetHitResult (int *query_selected_clusters, 
                                int *points_in_codebook_entry, 
                                int *points_in_codebook_entry_size, 
                                int *points_in_codebook_entry_bias, 
                                // uint8_t *hit_record, 
                                // uint8_t *hit_res, 
                                float *hit_record, 
                                float *hit_res, 
                                // thrust::pair<uint8_t, int> *hit_res, 
                                const int Q, const int nlists, const int D, const int M, const int PQ_entry) {
    int bit = threadIdx.x ;
    int bid = blockIdx.x ;
    int query = bid / (nlists * (D / M)) ;
    int nlist = bid / (D / M) % nlists;
    int dim = bid % (D / M);
    // printf ("device function\n") ;
    if (bit < PQ_entry && query < Q && nlist < nlists && dim < (D / M)) {
        __shared__ int cluster ;
        cluster = query_selected_clusters[query * nlists + nlist] ;
        __shared__ int size[32] ; // PQ_entry
        __shared__ int bias[32] ;
        __shared__ float record[32] ;
        size[bit] = points_in_codebook_entry_size[cluster * (D / M) * PQ_entry + dim * PQ_entry + bit] ;
        bias[bit] = points_in_codebook_entry_bias[cluster * (D / M) * PQ_entry + dim * PQ_entry + bit] ;
        record[bit] = hit_record[query * nlists * (D / M) * PQ_entry + nlist * (D / M) * PQ_entry + dim * PQ_entry + bit] ;
        __syncthreads() ;
        // printf ("%f\n", record[bit]) ;
        // int size = points_in_codebook_entry_size[cluster * (D / M) * PQ_entry + dim * PQ_entry + bit] ;
        // printf ("cluster %d, dim %d, bit %d, record %f\n", cluster, dim, bit, record[bit]) ;
        for (int i = 0; i < size[bit]; i ++) {
            int point = points_in_codebook_entry[bias[bit] + i] ;
            atomicAdd (hit_res + query * nlists * 3000 + nlist * 3000 + point, record[bit]) ;
        }
        __syncthreads() ;
    }
}

__global__ void gpuBucketSort (uint8_t *hit_res, int *bucket, int *bucket_ptr, int Q, int nlists) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int query = tid / (nlists * 3000) ;
    int nlist = tid / 3000 % nlists ;
    int point = tid % 3000 ;
    if (query < Q && nlist < nlists && point < 3000) {
        __shared__ uint8_t hit_res_shared[1000] ;
        __shared__ int bucket_ptr_shared[64] ;

        int point1000_idx = point / 1000 ;
        hit_res_shared[threadIdx.x] = hit_res[query * nlists * 3000 + nlist * 3000 + point1000_idx * 1000 + threadIdx.x] ;
        if (threadIdx.x < 64) {
            bucket_ptr_shared[threadIdx.x] = bucket_ptr[query * 64 + threadIdx.x] ;
        }
        __syncthreads() ;        

        int count = hit_res_shared[threadIdx.x] ;
        if (bucket_ptr_shared[count] < 100) {
            int ptr = atomicAdd(&bucket_ptr_shared[count], 1) ;
            bucket[query * 64 * 100 + count * 100 + ptr] = nlist * 3000 + point ;
        }
        if (threadIdx.x < 64) {
            bucket_ptr[query * 64 + threadIdx.x] = bucket_ptr_shared[threadIdx.x] ;
        }
        __syncthreads() ;
    }
}

__global__ void gpuTop100 (int *bucket, int *bucket_ptr, int *top100, int Q) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (qid < Q) {
        int cur_size = 0 ;
        for (int c = 0; c < 64; c ++) {
            for (int i = 0; i < min (100 - cur_size, bucket_ptr[qid * 64 + c]); i ++) {
                top100[qid * 100 + cur_size + i] = bucket[qid * 64 * 100 + c * 100 + i] ;
            }
            cur_size += min (100 - cur_size, bucket_ptr[qid * 64 + c]) ;
        }
    }
}

// query_selected_clusters: query * nlists
// points_in_codebook_entry: points_in_codebook_entry_total_size
// points_in_codebook_entry_size: C * (D / M) * PQ_entry
// points_in_codebook_entry_bias: C * (D / M) * PQ_entry
// d_hit_record: query * nlists * (D / M) * PQ_entry
float* getHitResult (int *query_selected_clusters, 
                    int *points_in_codebook_entry, 
                    int *points_in_codebook_entry_size, 
                    int *points_in_codebook_entry_bias, 
                    int points_in_codebook_entry_total_size, 
                    float *d_hit_record, 
                    int Q, int nlists, int C, int D, int M, int PQ_entry) {
    int *d_query_selected_clusters ;
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_query_selected_clusters), sizeof (int) * Q * nlists));
    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void*> (d_query_selected_clusters), query_selected_clusters, sizeof (int) * Q * nlists, cudaMemcpyHostToDevice));

    int *d_points_in_codebook_entry ;
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_points_in_codebook_entry), sizeof (int) * points_in_codebook_entry_total_size));
    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void*> (d_points_in_codebook_entry), points_in_codebook_entry, sizeof (int) * points_in_codebook_entry_total_size, cudaMemcpyHostToDevice));

    int *d_points_in_codebook_entry_size ;
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_points_in_codebook_entry_size), sizeof (int) * C * (D / M) * PQ_entry));
    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void*> (d_points_in_codebook_entry_size), points_in_codebook_entry_size, sizeof (int) * C * (D / M) * PQ_entry, cudaMemcpyHostToDevice));

    int *d_points_in_codebook_entry_bias ;
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_points_in_codebook_entry_bias), sizeof (int) * C * (D / M) * PQ_entry));
    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void*> (d_points_in_codebook_entry_bias), points_in_codebook_entry_bias, sizeof (int) * C * (D / M) * PQ_entry, cudaMemcpyHostToDevice));

    float *d_hit_res ;
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_hit_res), sizeof (float) * Q * nlists * 3000));
    CUDA_CHECK (cudaMemset (reinterpret_cast<void*> (d_hit_res), 0, sizeof (float) * Q * nlists * 3000));

    // thrust::pair<uint8_t, int> *d_hit_res ;
    // CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_hit_res), sizeof (thrust::pair<uint8_t, int>) * Q * nlists * 3000));

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st);
    int numOfThreads = Q * nlists * (D / M) * PQ_entry ;
    // printf("numOfThreads: %d\n", numOfThreads) ;
    dim3 block (numOfThreads / PQ_entry, 1), thread (PQ_entry, 1);
    printf ("Q: %d nlists: %d D: %d M: %d PQ_entry: %d\n", Q, nlists, D, M, PQ_entry);
    // printf ("%d %d %d\n", block.x, block.y, block.z) ;
    gpuGetHitResult<<<block, thread>>> (d_query_selected_clusters, d_points_in_codebook_entry, d_points_in_codebook_entry_size, d_points_in_codebook_entry_bias, d_hit_record, d_hit_res, Q, nlists, D, M, PQ_entry);
    CUDA_SYNC_CHECK();
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "GPU Hit Res: " << ms << std::endl;

    return d_hit_res ;

    // cudaEventRecord(st);
    // thrust::sort (thrust::device, d_hit_res, d_hit_res + Q * nlists * 3000, thrust::greater<thrust::pair<uint8_t, int>>()) ;
    // CUDA_SYNC_CHECK();
    // cudaEventRecord(ed);
    // cudaEventSynchronize(ed);
    // cudaEventElapsedTime(&ms, st, ed);
    // std::cout << "GPU Sort: " << ms << std::endl;

    // int *d_bucket ;
    // CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_bucket), sizeof (int) * Q * 64 * 100));
    // int *d_bucket_ptr ;
    // CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_bucket_ptr), sizeof (int) * Q * 64));
    // int *d_top100 ;
    // CUDA_CHECK (cudaMalloc (reinterpret_cast<void**> (&d_top100), sizeof (int) * Q * 100));

    // cudaEventRecord(st);
    // numOfThreads = Q * nlists * 3000 ;
    // block = dim3 (numOfThreads / 1000, 1), thread = dim3 (1000, 1);
    // gpuBucketSort<<<block, thread>>> (d_hit_res, d_bucket, d_bucket_ptr, Q, nlists);
    // CUDA_SYNC_CHECK();
    // numOfThreads = Q ;
    // block = dim3 (numOfThreads / 1024 + 1, 1), thread = dim3 (1024, 1);
    // gpuTop100<<<block, thread>>> (d_bucket, d_bucket_ptr, d_top100, Q);
    // CUDA_SYNC_CHECK();
    // cudaEventRecord(ed);
    // cudaEventSynchronize(ed);
    // cudaEventElapsedTime(&ms, st, ed);
    // std::cout << "GPU Bucket Sort: " << ms << std::endl;
}

}; // namespace juno