#ifndef JUNO_GPU_KERNEL_H_
#define JUNO_GPU_KERNEL_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>
namespace juno {

void plotQueryWithDensity(float* _search_points, float* _query, float* _centroids, int* _labels, int* _ground_truth, int _N, int _Q, int _D, int _C);

void referenceModel(float* _search_points, float* _query, float* _centroids, int* _labels, int* _ground_truth, int _N, int _Q, int _D, int _C, float** stat);

void counterOnGPU (std::vector<std::vector<std::pair<int, int>>> &query_cluster_mapping, int query_size, int nlists, int D, int M, int coarse_grained_cluster_num, 
    int cluster_bias[], std::vector<std::vector<int>> &cluster_query_mapping, 
    unsigned int *d_hit_record, int index_bias, 
    std::vector<int>*** inversed_codebook_map
) ;

void testHashmap () ;

}; // namespace juno

#endif