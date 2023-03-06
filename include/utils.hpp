#pragma once
#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <omp.h>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <optix.h>
#include <optix_host.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include "dbg.h"
#include "math.h"
#include "cblas.h"
#include <sys/time.h>
#include <assert.h>
#include "juno_gpu_kernel.cuh"

#define TRIANGLE_PER_HITABLE 6
#define HIT_MAGIC_NUMBER 114514
#define COARSE_GRAIN_CLUSTER_USE_GPU 0
#define NLISTS_MAX 1
#define PQ_DIM 2
#define SCALE 0.75
#define HIT_REC_PER_C 200
#define QUERY_BATCH_MAX 10000
#define VERBOSE 0

enum RT_MODE {
    QUERY_AS_RAY = 0,
    QUERY_AS_TRIANGLE = 1,
    RT_MODE_LEN = 2,
};

enum PRIMITIVE_TYPE {
    PRIMITIVE_TYPE_SPHERE = 0,
    PRIMITIVE_TYPE_TRIANGLE = 1,
    PRIMITIVE_TYPE_LEN = 2,
};

enum METRIC {
    METRIC_L2 = 0,
    METRIC_MIPS = 1,
    METRIC_COS = 2,
    METRIC_LEN = 3,
};

enum DATASET {
    SIFT1M = 0,
    SIFT1B = 1,
    TTI1M = 2,
    TTI1B = 3,
    DEEP1M = 4,
    CUSTOM = 5,
    DATASET_LEN = 6,
};

void elapsed(const char* evt, struct timeval st, struct timeval ed) {
    float res = (1000000.0 * (ed.tv_sec - st.tv_sec) + 1.0 * (ed.tv_usec - st.tv_usec));
    printf("[%32s]: %010.6fms\n", evt, res / 1000.0);
}

template <typename T>
T L2Dist(T* x, T* y, int D) {
    T res = 0.0;
    for (int i = 0; i < D; i++) {
        res += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(res);
}

template <typename T>
T L2Dist(std::vector<T> a, std::vector<T> b) {
    T res = 0.0;
    int D = a.size();
    for (int i = 0; i < D; i++) {
        res += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(res);
}

template <typename T>
void read_search_points(const char* path, T** _search_points, int _N, int _D) {
    std::ifstream fread_search_points(path, std::ios::in);
    for (int n = 0; n < _N; n++) {
        for (int d = 0; d < _D; d++) {
            fread_search_points >> _search_points[n][d];
        }
        fread_search_points.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    fread_search_points.close();
}

template <typename T>
void read_cluster_centroids(const char* path,T** _cluster_centroids, int _coarse_grained_cluster_num, int _D) {
    std::ifstream fread_cluster_centroids(path, std::ios::in);
    for (int c = 0; c < _coarse_grained_cluster_num; c++) {
        for (int d = 0; d < _D; d++) {
            fread_cluster_centroids >> _cluster_centroids[c][d];
        }
    }
    fread_cluster_centroids.close();
}

void read_search_points_labels(const char* path, int* _search_points_labels, int _N) {
    std::ifstream fread_search_points_labels(path, std::ios::in);
    for (int n = 0; n < _N; n++) {
        fread_search_points_labels >> _search_points_labels[n];
    }
    fread_search_points_labels.close();
}

template <typename T>
void read_queries(const char* path, T** _queries, int _Q, int _D) {
    std::ifstream fread_queries(path, std::ios::in);
    for (int q = 0; q < _Q; q++) {
        for (int d = 0; d < _D; d++) {
            fread_queries >> _queries[q][d];
        }
    }
    fread_queries.close();
}

void read_ground_truth(const char* path, int** _ground_truth, int _Q) {
    std::ifstream fread_ground_truth(path, std::ios::in);
    for (int q = 0; q < _Q; q++) {
        for (int gt = 0; gt < 100; gt++) {
            fread_ground_truth >> _ground_truth[q][gt];
        }
    }
    fread_ground_truth.close();
}

template <typename T>
void read_codebook_entry_labels(std::string path, T**** _codebook_entry, int*** _codebook_labels, int* _cluster_size, int _coarse_grained_cluster_num, int _PQ_entry, int _D, int _PQ_DIM=PQ_DIM) {
    std::string placeholder, labels;
    for (int c = 0; c < _coarse_grained_cluster_num; c++) {
        for (int d = 0; d < _D / _PQ_DIM; d++) {
            std::ifstream fread_codebook((path + "/codebook_cluster=" + std::to_string(c) + "_dim=" + std::to_string(d)).c_str(), std::ios::in);
            for (int e = 0; e < _PQ_entry; e++) {
                for (int m = 0; m < _PQ_DIM; m++) {
                    fread_codebook >> _codebook_entry[c][d][e][m];
                }
            }
            fread_codebook >> placeholder;
            for (int l = 0; l < _cluster_size[c]; l++) {
                fread_codebook >> _codebook_labels[c][d][l];
            }
            fread_codebook.close();
        }
    }
}


#endif