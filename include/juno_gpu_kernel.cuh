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

void getHitResult (int *query_selected_clusters, 
                    int *points_in_codebook_entry, 
                    int *points_in_codebook_entry_size, 
                    int *points_in_codebook_entry_bias, 
                    int points_in_codebook_entry_total_size, 
                    float *d_hit_record, 
                    int Q, int nlists, int C, int D, int M, int PQ_entry) ;
}; // namespace juno

#endif