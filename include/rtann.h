#ifndef RTANN_H_
#define RTANN_H_
#include <optix.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define MAGIC_NUMBER_0 114514
#define MAGIC_NUMBER_1 1919810
#define N_RESULT 1024
#define N_RECALL 128
#define USE_EMPRICAL_THRES 1
#define SEARCH_AS_PRIMITIVES 1

struct Params {
    OptixTraversableHandle handle;
};

struct RayGenData {
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
    float3* ray_origin;
#if SEARCH_AS_PRIMITIVES == 0
    unsigned int* ray_hit;
#else
    unsigned int* prim_hit;
#endif
};

struct MissData {
    float r, g, b;
};

struct HitGroupData {
    unsigned int* prim_hit;
    // No data needed
};

namespace rtann {
void search(      float**   queries,            /* NQ * D */
            const int&      NQ, 
            const int&      D,
                  float***  codebook,           /* 2^nbits * M */
            const int&      M,                  /* D / 2 == M */
            const int&      nbits, 
            const float*    dist_thres,
            const float     dist_thres_scale,
            // const bool&     search_as_triangle,
                  float*    distance,
                  float**   index,
                  std::vector<std::vector<float>>& res);

void load_query(const char* query_path, const int& NQ, const int& D, float** queries);
void load_codebook(const char* codebook_path, const int& M, const int& nbits, float*** codebook, float* dist_medium);
}; // namespace rtann
#endif
