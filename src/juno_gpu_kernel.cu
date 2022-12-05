#include "cuda.h"
#include "juno_gpu_kernel.cuh"
#include <stdio.h>
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


}; // namespace juno