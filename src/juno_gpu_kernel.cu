#include "cuda.h"
#include "juno_gpu_kernel.cuh"
#include <stdio.h>
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

__global__ void vanillaRef(float* p, float* q, float* c, int* l, int* gt, float* res, int _N, int_Q, int _D, int _C) {

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
    h_res = new int[_Q];
    h_l2 = new float[_N];
    h_r = new float[_Q * _D / 2];
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_search_points), sizeof(float) * _N * _D));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query), sizeof(float) * _Q * _D));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centroids), sizeof(float) * _C * _D));
    CUDA_CHECK(cudaMalloc(reinterpret_castMvoid**>(&d_labels), sizeof(int) * _N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ground_truth), sizeof(int) * _Q * 100));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_res), sizeof(int) * _Q));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l2), sizeof(float) * _N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_r), sizeof(float) * _Q * _D / 2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_recall), sizeof(float) * _Q));
    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_search_points), _search_points, sizeof(float) * _N * _D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_query), _query, sizeof(float) * _Q * _D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ground_truth), _ground_truth, sizeof(int) * _Q * 100, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centroids), _centroids, sizeof(float) * _C * _D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_labels), _labels, sizeof(int) * _N, cudaMemcpyHostToDevice));
    
    vanillaRef<<<>>>(d_search_points, d_query, d_centroids, d_labels, d_ground_truth, d_recall, _N, _Q, _D, _C);
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