#pragma once
#ifndef JUNO_RT_H_
#define JUNO_RT_H_

#include "utils.hpp"

struct Params {
    OptixTraversableHandle handle;
    int nlists, dim, bit ;
    float radius ;
};

struct RayGenData {
    // float3 cam_eye;
    // float3 camera_u, camera_v, camera_w;
    float3* ray_origin;
    // unsigned int* ray_hit;
};

struct MissData {
    float r, g, b;
};

struct HitGroupData {
    // unsigned int* hit_record;
    // int *hit_record ; // query * nlists * (D / M) * PQ_entry
    float *hit_record ;
    // int *query_selected_clusters ; // query * nlists
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}


namespace juno {

template <typename T>
class juno_rt {
private:
    OptixDeviceContext              context             = nullptr;
    OptixDeviceProperty             property            = {};
    CUcontext                       cuCtx               = 0;
    OptixDeviceContextOptions       options             = {};

    OptixTraversableHandle          gas_handle;
    CUdeviceptr                     d_gas_output_buffer;
    CUdeviceptr                     d_params;
    OptixAccelBuildOptions          accel_options       = {};

    OptixPipeline                   pipeline            = nullptr;
    OptixShaderBindingTable         sbt                 = {};
    char                            log[2048];

    float3*                         d_ray_origin;
    float3*                         d_ray_origin_whole;
    Params                          params;
    // unsigned int*                   d_hit_record;
    // int*                        d_hit_record;
    float*                             d_hit_record;
    // int*                            d_query_selected_clusters;
    int                             hitable_num;
    int                             dim_pair;

    float factors[64] = {226.91, 226.292, 234.105, 245.577, 279.63, 236.516, 231.948, 269.431, 274.614, 244.002, 235.553, 258.38, 243.939, 237.857, 229.811, 229.819, 244.322, 226.982, 252.21, 246.903, 265.966, 238.008, 231.935, 249.658, 278.304, 241.357, 236.966, 259.187, 245.247, 245.449, 244.663, 229.863, 238.673, 245.904, 235.468, 238.296, 266.595, 246.564, 229.863, 245.392, 275.224, 245.247, 239.019, 254.136, 239.708, 236.212, 248.244, 244.125, 237.346, 247.491, 225.754, 225.657, 276.957, 235.85, 229.142, 265.548, 285.272, 237.186, 252.723, 263.139, 240.983, 220.048, 237.626, 236.326};
        
public:
    juno_rt(int _Q=10000, int _D=128) {
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin), sizeof(float3) * 64 * 80));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin_whole), sizeof(float3) * _Q * (_D / 2) * NLISTS_MAX));
        cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params));
    }

    void constructCompleteBVHwithPQ(T**** _codebook_entry,
                                    int _coarse_grained_cluster_num,
                                    int _C, int _D, int _M,
                                    T** _stat,
                                    T _r,
                                    METRIC _metric) 
    {
        struct timeval st, ed;
        float3* centers;
        float* radius;
        dim_pair = _D / _M;
        int prim_idx = 0;
        int num_sphere_per_dim_pair = _C;
        hitable_num = _coarse_grained_cluster_num * num_sphere_per_dim_pair * dim_pair;
        centers = new float3[hitable_num];
        radius = new float[hitable_num];
        for (int c = 0; c < _coarse_grained_cluster_num; c++) {
            for (int d = 0; d < dim_pair; d++) {
                // float _radius = _r * factors[d];
                float _radius = _r * 1.0;
                for (int n = 0; n < num_sphere_per_dim_pair; n++) {
                    float x = (1.0 * _codebook_entry[c][d][n][0]) / 100.0;
                    float y = (1.0 * _codebook_entry[c][d][n][1]) / 100.0;
                    // float factor = 0.01 * std::min(x, y);
                    // if (c == 432) {
                    //     printf("Prim %d, c %d, d %d, bit %d: (%.6f, %.6f, %.6f)\n", prim_idx, c, d, n, x, y, 1.0 * (c * 128 + 2 * d + 1));
                    // }
                    centers[c * dim_pair * num_sphere_per_dim_pair + d * num_sphere_per_dim_pair + n] = make_float3(x, y, 1.0 * (c * 128 + 2 * d + 1));
                    // if (c == 432 && d == 0) printf("Prim %d:(%.6f, %.6f, %.6f)\n", prim_idx, x, y, 1.0 * (c * 128 + 2 * d + 1));
                    // radius[c * dim_pair * num_sphere_per_dim_pair + d * num_sphere_per_dim_pair + n] = static_cast<float>(0.45 + factor);
                    radius[c * dim_pair * num_sphere_per_dim_pair + d * num_sphere_per_dim_pair + n] = static_cast<float>(_radius);
                    prim_idx++;
                }
            }
        }

        dbg (hitable_num) ;

        CUdeviceptr d_centers;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centers), hitable_num * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centers), centers, hitable_num * sizeof(float3), cudaMemcpyHostToDevice));
    
        CUdeviceptr d_radius;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius), hitable_num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius), radius, hitable_num * sizeof(float), cudaMemcpyHostToDevice));

        uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput sphere_input = {};
        sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphere_input.sphereArray.vertexBuffers = &d_centers;
        sphere_input.sphereArray.numVertices = hitable_num;
        sphere_input.sphereArray.radiusBuffers = &d_radius;
        sphere_input.sphereArray.flags = sphere_input_flags;
        sphere_input.sphereArray.numSbtRecords = 1;

        gettimeofday(&st, NULL);
        constructBVHTreeWithPrimitives(&sphere_input, PRIMITIVE_TYPE_SPHERE);
        gettimeofday(&ed, NULL);
        elapsed("Constructing BVH", st, ed);
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_centers)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_radius)));
    }

    // Construct a big whole BVH and hit with tag.
    // At most 256 clusters are supported, how to handle 1B dataset?
    void constructBVHwithPQ(int _label,             // BVH build for _label(th) cluster
                            T**** _codebook_entry,  // use _codebook_entry[_label][...][...][...]
                            int _C,                 // PQ_entry
                            int _D,                 // Dimension
                            int _M,                 // == 2
                            T** _stat,              // min(d), max(d)
                            T _r, 
                            METRIC _metric
                            ) 
    {
        float3* centers;
        float* radius;
        int num_circles_per_dimpair = _C;
        dim_pair = _D / _M;
        hitable_num = num_circles_per_dimpair * dim_pair;
        centers = new float3[hitable_num];
        radius = new float[hitable_num];
        for (int d = 0; d < dim_pair; d++) {
            float _radius = _r * factors[d];
            for (int n = 0; n < num_circles_per_dimpair; n++) {
                float x = 1.0 * _codebook_entry[_label][d][n][0];
                float y = 1.0 * _codebook_entry[_label][d][n][1];
                centers[d * num_circles_per_dimpair + n] = make_float3(x, y, 1.0 * (2 * d + 1));
                radius[d * num_circles_per_dimpair + n] = _radius;
            }
        }
        CUdeviceptr d_centers;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centers), hitable_num * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centers), centers, hitable_num * sizeof(float3), cudaMemcpyHostToDevice));
    
        CUdeviceptr d_radius;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius), hitable_num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius), radius, hitable_num * sizeof(float), cudaMemcpyHostToDevice));

        uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput sphere_input = {};
        sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphere_input.sphereArray.vertexBuffers = &d_centers;
        sphere_input.sphereArray.numVertices = hitable_num;
        sphere_input.sphereArray.radiusBuffers = &d_radius;
        sphere_input.sphereArray.flags = sphere_input_flags;
        sphere_input.sphereArray.numSbtRecords = 1;

        constructBVHTreeWithPrimitives(&sphere_input, PRIMITIVE_TYPE_SPHERE);
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_centers)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_radius)));

    }

    void constructBVHforLabelWithRadius(int _label, T** _search_points, int* _search_points_labels, int _N, int _D, T** _stat, T _radius, METRIC _metric) {
        float3* vertices;
        int num_vertices;
        std::vector <int> point_index_of_label;
        for (int i = 0; i < _N; i++) {
            if (_search_points_labels[i] == _label) {
                point_index_of_label.push_back(i);
            }
        }
        int M;
        switch (_metric) {
            case METRIC_L2: {
                M = 2;
                dim_pair = _D / M;
                hitable_num = point_index_of_label.size(); 
                hitable_num = 64;
                num_vertices = hitable_num * TRIANGLE_PER_HITABLE * dim_pair;
                vertices = new float3[num_vertices];
                for (int d = 0; d < dim_pair; d++) {
                    // T a = _stat[d*2][1], b = _stat[d*2+1][1];
                    // T factor = sqrt(a * a + b * b);
                    // float factor = std::sqrt(_stat[d << 1][1] * _stat[d << 1][1] + _stat[d << 1 + 1][1] * _stat[d << 1 + 1][1]);
                    _radius = 0.25 * factors[d];
                    for (int n = 0; n < hitable_num; n++) {
                        T x = _search_points[point_index_of_label[n]][(d << 1)];
                        T y = _search_points[point_index_of_label[n]][(d << 1) + 1];
                        float3 v1 = make_float3(x - _radius, y - _radius, 1.0 * d + 1);
                        float3 v2 = make_float3(x - _radius, y + _radius, 1.0 * d + 1);
                        float3 v3 = make_float3(x + _radius, y + _radius, 1.0 * d + 1);
                        float3 v4 = make_float3(x - _radius, y - _radius, 1.0 * d + 1);
                        float3 v5 = make_float3(x + _radius, y + _radius, 1.0 * d + 1);
                        float3 v6 = make_float3(x + _radius, y - _radius, 1.0 * d + 1);
                        int base_idx = d * hitable_num * TRIANGLE_PER_HITABLE + n * TRIANGLE_PER_HITABLE;
                        vertices[base_idx + 0] = v1;
                        vertices[base_idx + 1] = v2;
                        vertices[base_idx + 2] = v3;
                        vertices[base_idx + 3] = v4;
                        vertices[base_idx + 4] = v5;
                        vertices[base_idx + 5] = v6;
                    }
                }
                CUdeviceptr d_vertices;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), num_vertices * sizeof(float3)));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices, num_vertices * sizeof(float3), cudaMemcpyHostToDevice));

                const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
                OptixBuildInput triangle_input = {};
                triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                triangle_input.triangleArray.numVertices = static_cast<uint32_t>(num_vertices);
                triangle_input.triangleArray.vertexBuffers = &d_vertices;
                triangle_input.triangleArray.flags = triangle_input_flags;
                triangle_input.triangleArray.numSbtRecords = 1;
                
                constructBVHTreeWithPrimitives(&triangle_input, PRIMITIVE_TYPE_TRIANGLE);
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
                break;
            }
            case METRIC_MIPS:
                M = 3;
                break;
            case METRIC_COS:
                M = 3;
                break;
            default:
                M = -1;
                break;
        }

    }

    void constructBVHTreeWithPrimitives(OptixBuildInput* primitive_input, PRIMITIVE_TYPE prim_type) {
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, primitive_input, 1, &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, primitive_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
                
        OptixModule module = nullptr;
        OptixModule triangle_module;
        OptixModuleCompileOptions module_compile_options = {};
        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 3;
        pipeline_compile_options.numAttributeValues = 3;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = (prim_type == PRIMITIVE_TYPE_SPHERE) ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
                
        std::string input;
        std::ifstream file(std::getenv("JUNO_ROOT") + std::string("/src/juno_rt.optixir"), std::ios::binary);
        if (file.good()) {
            std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
            input.assign(buffer.begin(), buffer.end());
        }

        size_t inputSize = input.size();
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(context, &module_compile_options, &pipeline_compile_options, input.c_str(), inputSize, log, &sizeof_log, &module));
        OptixBuiltinISOptions builtin_is_options = {};
        builtin_is_options.usesMotionBlur = false;

        builtin_is_options.builtinISModuleType = (prim_type == PRIMITIVE_TYPE_SPHERE) ? OPTIX_PRIMITIVE_TYPE_SPHERE : OPTIX_PRIMITIVE_TYPE_TRIANGLE;
        OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options, &builtin_is_options, &triangle_module));

        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;

        OptixProgramGroupOptions program_group_options = {};
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                                &raygen_prog_group_desc, 
                                                1, 
                                                &program_group_options,
                                                log,
                                                &sizeof_log,
                                                &raygen_prog_group
                                                ));
                
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                                &miss_prog_group_desc, 
                                                1, 
                                                &program_group_options, 
                                                log, 
                                                &sizeof_log, 
                                                &miss_prog_group
                                                ));
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hitgroup_prog_group_desc.hitgroup.moduleAH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        hitgroup_prog_group_desc.hitgroup.moduleIS = triangle_module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                                &hitgroup_prog_group_desc, 
                                                1, 
                                                &program_group_options, 
                                                log, 
                                                &sizeof_log, 
                                                &hitgroup_prog_group
                                                ));
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};
        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(context, 
                                            &pipeline_compile_options, 
                                            &pipeline_link_options, 
                                            program_groups, 
                                            sizeof(program_groups) / sizeof(program_groups[0]), 
                                            log, 
                                            &sizeof_log, 
                                            &pipeline
                                            ));
        OptixStackSizes stack_sizes = {};
        for (auto& prog_group: program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_satck_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 
                                                max_trace_depth, 
                                                0, 
                                                0, 
                                                &direct_callable_stack_size_from_traversal, 
                                                &direct_callable_stack_size_from_state, 
                                                &continuation_satck_size
                                                ));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 
                                              direct_callable_stack_size_from_traversal, 
                                                direct_callable_stack_size_from_state, 
                                                continuation_satck_size, 
                                                1
                                                ));

        // data of d_ray_origin should be feed runtime.
        CUdeviceptr raygen_record;
        size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        rg_sbt.data.ray_origin = d_ray_origin_whole;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));
        sbt.raygenRecord = raygen_record;

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        ms_sbt.data = {0.0f, 0.0f, 0.0f};
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;  
                
        // int d_hit_record_size = QUERY_BATCH_MAX * dim_pair * NLISTS_MAX;
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), sizeof(unsigned int) * d_hit_record_size));
        // HARDCODE, query * nlists * dim * bit
        int hit_record_size = 10000 * 64 * 96 * 32 ;
        // int d_hit_record_size = 1 * 8 * 64 * 32 ;
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), sizeof(uint8_t) * d_hit_record_size)) ;
        // float *hit_record = new float[hit_record_size] ;
        // std::fill (hit_record, hit_record + hit_record_size, -1.0f) ; // HARDCODE, -dim
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), sizeof(float) * hit_record_size)) ;
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_hit_record), 0, sizeof(float) * hit_record_size)) ;
        // CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit_record), hit_record, sizeof(float) * hit_record_size, cudaMemcpyHostToDevice)) ;
        // int d_query_selected_clusters_size = 10000 * 8 ; // query * nlists
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_selected_clusters), sizeof(int) * d_query_selected_clusters_size)) ;

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        hg_sbt.data.hit_record = d_hit_record;
        // hg_sbt.data.query_selected_clusters = d_query_selected_clusters ;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1; 

        params.handle = gas_handle;
        // params.nlists = 8, params.dim = 64, params.bit = 32; // HARDCODE
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));
    }

    void setParams (int nlists, int dim, int bit, int radius) {
        params.nlists = nlists, params.dim = dim, params.bit = bit;
        params.radius = radius ;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));
    }

    void setRayOrigin(float3* ray_origin, int size) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ray_origin_whole), ray_origin, sizeof(float3) * size, cudaMemcpyHostToDevice));
    }

    void getRayHitRecord(unsigned int* hit_record, int size) {
        CUDA_CHECK(cudaMemcpy(hit_record, reinterpret_cast<void*>(d_hit_record), sizeof(unsigned int) * size, cudaMemcpyDeviceToHost));
    }

    // void setQuerySelectedClusters (int* query_selected_clusters, int size) {
    //     CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_query_selected_clusters), query_selected_clusters, sizeof(int) * size, cudaMemcpyHostToDevice));
    // }

    // static void initRayOriginArray(int Q, int D, int M, int _nlists) {
    //     CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin), sizeof(float3) * Q * _nlists * D / M));
    // }

    OptixPipeline& getOptixPipeline() {
        return pipeline;
    }

    CUdeviceptr getDparams() {
        return d_params;
    }

    OptixShaderBindingTable* getSBT() {
        return &sbt;
    }

    float* getPrimitiveHit() {
        return d_hit_record;
    }

    OptixTraversableHandle& getGasHandle() {
        return gas_handle;
    }

    int getHitableNum() {
        return hitable_num * dim_pair;
    }
}; // class juno_rt

}; // namespace juno

#endif