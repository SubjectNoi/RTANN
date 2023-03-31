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
    const float3 origin = rgData->ray_origin[idx.x];
    // const float3 origin = make_float3(1.0, 2.0, 3.0);
    // if (idx.x == 0) printf("Ray:%d Generated, Origin:(%f,%f,%f)\n", idx.x, origin.x, origin.y, origin.z);
    optixTrace(params.handle, 
               origin, 
               direction, 
               0.0f, 
               1.0f, 
               0.0f, 
               OptixVisibilityMask(1), 
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
    // const float t = optixGetRayTmax();
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    HitGroupData* htData = (HitGroupData*)optixGetSbtDataPointer();
    unsigned long long one = 1;
    htData->hit_record[idx.x * (MAX_ENTRY / 32) + (((prim_idx / 2) % 32) / 32)] |= (one << (prim_idx % 64));
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