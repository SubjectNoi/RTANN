#include <optix.h>
#include "rtann.h"
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
    const float3 origin = rgData->ray_origin[idx.x];
    unsigned int hit;
    unsigned int hit_primitive_id = 0;
    // printf("Ray:%d Generated, Origin:(%f,%f,%f)\n", idx.x, origin.x, origin.y, origin.z);
    optixTrace(params.handle, 
               origin, 
               direction, 
               0.0f, 
               0.75f, 
               0.0f, 
               OptixVisibilityMask(1), 
               OPTIX_RAY_FLAG_NONE, 
               0, 
               0, 
               0,
               hit,
               hit_primitive_id
              );
#if SEARCH_AS_PRIMITIVES == 0
    rgData->ray_hit[idx.x] = hit;
#else
    rgData->prim_hit[hit_primitive_id] = MAGIC_NUMBER_1;
#endif
}

extern "C" __global__ void __miss__ms() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
}

extern "C" __global__ void __anyhit__ah() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    HitGroupData* htData = (HitGroupData*)optixGetSbtDataPointer();

    // Wierd, this raise error: lvalue can't be modified, but in ray gen there is a const, now we just abandon const.
    // const HitGroupData* htData = (HitGroupData*)optixGetSbtDataPointer();
#if SEARCH_AS_PRIMITIVES == 0
    optixSetPayload_0(MAGIC_NUMBER_0);
#else
    optixSetPayload_1(prim_idx >> 1);
    htData->prim_hit[prim_idx>>1] = 114514;
#endif
    optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__ch() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();
    // printf("Ray:%d Hit (Closest), whose origin is: (%f,%f,%f) --> (%f,%f,%f | %f)\n", idx.x, ray_orig.x, ray_orig.y, ray_orig.z, ray_dir.x, ray_dir.y, ray_dir.z, t);
    float3 hit_point = make_float3(ray_orig.x + t * ray_dir.x, ray_orig.y + t * ray_dir.y, ray_orig.z + t * ray_dir.z);
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
