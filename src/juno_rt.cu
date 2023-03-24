#include <optix.h>
#include "juno_rt.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
    __constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    const RayGenData* rgData = (RayGenData*)optixGetSbtDataPointer();
    // printf("Getting ray origin of %d\n", idx.x);
    int ray_idx = idx.x * dim.y * dim.z + idx.y * dim.z + idx.z ;
    const float3 origin = rgData->ray_origin[ray_idx];
    // const float3 origin = make_float3(1.0, 2.0, 3.0);
    // printf("Ray:%d Generated, Origin:(%f,%f,%f)\n", idx.x, origin.x, origin.y, origin.z);
    optixTrace(params.handle, 
               origin, 
               direction, 
               0.0f, 
               1.0f, 
               0.0f, 
               OptixVisibilityMask(255), 
               OPTIX_RAY_FLAG_NONE, 
               0, 
               0, 
               0
              );
}

extern "C" __global__ void __miss__ms() {
    // const uint3 idx = optixGetLaunchIndex();
    // const uint3 dim = optixGetLaunchDimensions();
    // printf("Miss!: %d\n", idx.x);
}

extern "C" __global__ void __anyhit__ah() {
    const uint3 idx = optixGetLaunchIndex();
    // const uint3 dim = optixGetLaunchDimensions();
    // const float3 ray_orig = optixGetWorldRayOrigin();
    // const float3 ray_dir = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    HitGroupData* htData = (HitGroupData*)optixGetSbtDataPointer();
    int query = idx.x, index = idx.y, dim = idx.z ;
    // int cluster = prim_idx / (params.dim * params.bit) ;
    // int dim = prim_idx % (64 * 32) / 32;
    int bit = prim_idx % params.bit;
    // int index = 0 ;
    // for (int i = 0; i < 8; i ++) { // nlists
    //     if (htData -> query_selected_clusters[query * 8 + i] == cluster) {
    //         index = i ;
    //         break ;
    //     }
    // }
    // printf ("query:%d index:%d cluster:%d dim:%d bit:%d prim_idx: %d\n", query, index, cluster, dim, bit, prim_idx) ;
    // float dis = 0.5 * 0.5 - (1 - t) * (1 - t) ; // d^2 = r^2 - (1 - t)^2, HARDCODE
    float dis = 100.0 * params.radius * 100 * params.radius - (100.0 - 100.0 * t) * (100.0 - 100.0 * t) ;
    htData -> hit_record[query * (params.nlists * params.dim * params.bit) + index * (params.dim * params.bit) + dim * params.bit + bit] += 10000.0 - dis ;
    // htData -> hit_record[query * (params.nlists * params.dim * params.bit) + index * (params.dim * params.bit) + dim * params.bit + bit] += 1 ;
    // unsigned int one = 1;
    // htData->hit_record[idx.x] |= (one << (prim_idx % 32));
    // htData->prim_hit[prim_idx>>1] = 114514;
    // htData->prim_hit[prim_idx >> 1] = 114514;
    optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__ch() {
    // const uint3 idx = optixGetLaunchIndex();
    // const uint3 dim = optixGetLaunchDimensions();
    // const float3 ray_orig = optixGetWorldRayOrigin();
    // const float3 ray_dir = optixGetWorldRayDirection();
    // const float t = optixGetRayTmax();
    // printf("Ray:%d Hit (Closest), whose origin is: (%f,%f,%f) --> (%f,%f,%f | %f)\n", idx.x, ray_orig.x, ray_orig.y, ray_orig.z, ray_dir.x, ray_dir.y, ray_dir.z, t);
    // float3 hit_point = make_float3(ray_orig.x + t * ray_dir.x, ray_orig.y + t * ray_dir.y, ray_orig.z + t * ray_dir.z);
    // optixTrace(params.handle, 
    //     hit_point, 
    //     ray_dir, 
    //     0.0f, 
    //     1e16f, 
    //     0.0f, 
    //     OptixVisibilityMask(1), 
    //     OPTIX_RAY_FLAG_NONE, 
    //     0, 
    //     2, 
    //     0
    //    );
}