#ifndef RTANN_H_
#define RTANN_H_
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sys/resource.h>
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

class rtann {
private:
    CUcontext cuCtx = 0;
    CUdeviceptr d_gas_output_buffer;
    OptixDeviceContext context = nullptr;
    OptixTraversableHandle gas_handle;
    OptixDeviceContextOptions options = {};
    OptixAccelBuildOptions accel_options = {};

    int n_points, n_dim, n_queries;
    int n_clusters, n_codebook, n_codebook_idx_bitwidth, n_codebook_entry;
    char *points_file_path;
public:
    
    // utils: load_points(), load_queries()
    // init()
    // ivf_construct(), ivf_search()
    // codebook_construct(), bvh_construct_global(), bvh_construct_local(), bvh_construct_query()
    // bvh_search(), calculate_orrurance()
    // build_index()
    // search()

    /* build_index(ivf, bvh_global);
     * search(q, query_as_ray, global_bvh) {
     *     ivf_search();
     *     calculate_occurance();
     *     // Sort and Return 
     * }
     * 
     * 
     * build_index(ivf);
     * search(q, query_as_primitive, query_bvh) {
     *     ivf_search();
     *     bvh_construct_query();
     *     bvh_search();
     *     calculate_occurance();
     * }
     */

};

}; // namespace rtann

#endif
