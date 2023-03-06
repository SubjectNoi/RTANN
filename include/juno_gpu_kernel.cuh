#ifndef JUNO_GPU_KERNEL_H_
#define JUNO_GPU_KERNEL_H_

#include "cuda.h"
#include "cuda_runtime.h"
// #include "utils.hpp"
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

void getHitResult(unsigned int* _hit_record, 
                  uint8_t* _hit_res, 
                  int _nlists,
                  int* _all_candidates,
                  int* _all_candidates_cluster,
                  int* _all_candidates_bias,
                  unsigned int* _candidates_belong_on_every_dim);
                //   std::vector<std::vector<std::pair<int, int>>> _query_cluster_mapping,
                //   int* _cluster_bias,
                //   std::vector<std::vector<int>> _cluster_query_mapping,
                //   int* _cluster_query_size,
                //   std::vector<int>*** _inversed_codebook_map,
                //   int* _sub_cluster_size);

}; // namespace juno

#endif