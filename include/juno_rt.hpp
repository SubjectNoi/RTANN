#pragma once
#ifndef JUNO_RT_H_
#define JUNO_RT_H_

#include "utils.hpp"

struct Params {
    OptixTraversableHandle handle;
    int visibilityMask ;
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
    unsigned int* prim_hit;
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
    Params                          params;
    unsigned int*                   d_hit;
    int                             hitable_num;
    int                             dim_pair;
public:
    juno_rt() {
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin), sizeof(float3) * 64));

        // dbg (OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK) ;
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
                // constructing 1st group of triangles
                M = 2;
                dim_pair = _D / M;
                hitable_num = point_index_of_label.size(); 
                // hitable_num = 256;
                num_vertices = hitable_num * TRIANGLE_PER_HITABLE * dim_pair;
                vertices = new float3[num_vertices];
                for (int d = 0; d < dim_pair; d++) {
                    T a = _stat[d*2][1], b = _stat[d*2+1][1];
                    T factor = sqrt(a * a + b * b);
                    // float factor = std::sqrt(_stat[d << 1][1] * _stat[d << 1][1] + _stat[d << 1 + 1][1] * _stat[d << 1 + 1][1]);
                    _radius = 0.2 * factor;
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

                // OptixBuild 1st group of triangles              
                const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
                OptixBuildInput triangle_input = {};
                triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                triangle_input.triangleArray.numVertices = static_cast<uint32_t>(num_vertices);
                triangle_input.triangleArray.vertexBuffers = &d_vertices;
                triangle_input.triangleArray.flags = triangle_input_flags;
                triangle_input.triangleArray.numSbtRecords = 1;

                OptixAccelBufferSizes gas_buffer_sizes;
                OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));
                CUdeviceptr d_temp_buffer_gas;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
                OptixTraversableHandle triangleHandle = 0 ;
                OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &triangleHandle, nullptr, 0));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));

                // construct 2nd group of triangles
                float3* testVisibleMaskVertices = new float3[3];
                testVisibleMaskVertices[0] = make_float3 (0, 0, 0) ;
                testVisibleMaskVertices[1] = make_float3 (1, 0, 0) ;
                testVisibleMaskVertices[2] = make_float3 (0, 1, 0) ;
                CUdeviceptr d_testVertices;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_testVertices), 3 * sizeof(float3)));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_testVertices), testVisibleMaskVertices, 3 * sizeof(float3), cudaMemcpyHostToDevice));

                // OptixBuild 2nd group of triangles
                OptixBuildInput test_triangle_input = {};
                test_triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                test_triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                test_triangle_input.triangleArray.numVertices = static_cast<uint32_t>(3);
                test_triangle_input.triangleArray.vertexBuffers = &d_testVertices;
                test_triangle_input.triangleArray.flags = triangle_input_flags;
                test_triangle_input.triangleArray.numSbtRecords = 1;

                OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &test_triangle_input, 1, &gas_buffer_sizes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
                // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
                OptixTraversableHandle testTriangleHandle = 0 ;
                OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &test_triangle_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &testTriangleHandle, nullptr, 0));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_testVertices)));

                // two groups of triangles as instances
                // OptixAccelBuildOptions accelOptions = {};
                OptixBuildInput buildInput ;
                // memset(reinterpret_cast<void*>(buildInput), 0, sizeof (OptixBuildInput)) ;

                // memset(reinterpret_cast<void*>(&accelOptions), 0, sizeof(OptixAccelBuildOptions));
                // accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
                // accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
                // accelOptions.motionOptions.numKeys = 0;

                OptixInstance instances[2] ;

                // 1st instance
                OptixInstance* triangleInstance = &instances[0] ;
                triangleInstance -> instanceId = 0;
                // set instance visibilityMask
                triangleInstance -> visibilityMask = 1;
                triangleInstance -> sbtOffset = 0;
                triangleInstance -> flags = OPTIX_INSTANCE_FLAG_NONE;
                triangleInstance -> traversableHandle = triangleHandle ;

                // 2nd instance
                OptixInstance* testTriangleInstance = &instances[1] ;
                testTriangleInstance -> instanceId = 1;
                // set instance visibilityMask
                testTriangleInstance -> visibilityMask = 2;
                testTriangleInstance -> sbtOffset = 0;
                testTriangleInstance -> flags = OPTIX_INSTANCE_FLAG_NONE;
                testTriangleInstance -> traversableHandle = testTriangleHandle ;

                
                CUdeviceptr d_instance ;
                CUDA_CHECK (cudaMalloc (reinterpret_cast<void**>(&d_instance), sizeof (OptixInstance) * 2)) ;
                CUDA_CHECK (cudaMemcpy (reinterpret_cast<void*>(d_instance), &instances, sizeof (OptixInstance) * 2, cudaMemcpyHostToDevice)) ;

                // OptixBuild the whole handle
                OptixBuildInputInstanceArray* instanceArray = &buildInput.instanceArray;
                buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES ;
                instanceArray -> instances = d_instance;
                instanceArray -> numInstances = 2;

                OptixAccelBufferSizes bufferSizes = {};
                OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
                    &buildInput, 1, &bufferSizes));

                CUdeviceptr d_output, d_temp;

                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bufferSizes.outputSizeInBytes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes));

                OPTIX_CHECK(optixAccelBuild(context, 0,
                    &accel_options, &buildInput, 1, d_temp,
                    bufferSizes.tempSizeInBytes, d_output,
                    bufferSizes.outputSizeInBytes, &gas_handle, nullptr, 0));
                
                OptixModule module = nullptr;
                OptixModule triangle_module;
                OptixModuleCompileOptions module_compile_options = {};
                OptixPipelineCompileOptions pipeline_compile_options = {};
                pipeline_compile_options.usesMotionBlur = false;
                pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
                pipeline_compile_options.numPayloadValues = 3;
                pipeline_compile_options.numAttributeValues = 3;
                pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
                pipeline_compile_options.usesPrimitiveTypeFlags = 0;
                
                std::string input;
                std::ifstream file("/home/wtni/RTANN/RTANN/src/juno_rt.optixir", std::ios::binary);
                if (file.good()) {
                    std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
                    input.assign(buffer.begin(), buffer.end());
                }

                size_t inputSize = input.size();
                size_t sizeof_log = sizeof(log);
                OPTIX_CHECK_LOG(optixModuleCreateFromPTX(context, &module_compile_options, &pipeline_compile_options, input.c_str(), inputSize, log, &sizeof_log, &module));
                OptixBuiltinISOptions builtin_is_options = {};
                builtin_is_options.usesMotionBlur = false;
                builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
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
                                                    2
                                                    ));

                // data of d_ray_origin should be feed runtime.
                CUdeviceptr raygen_record;
                size_t raygen_record_size = sizeof(RayGenSbtRecord);
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
                RayGenSbtRecord rg_sbt;
                rg_sbt.data.ray_origin = d_ray_origin;
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
                
                int d_hit_size = hitable_num * dim_pair;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit), sizeof(unsigned int) * d_hit_size));
                CUdeviceptr hitgroup_record;
                size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
                HitGroupSbtRecord hg_sbt;
                hg_sbt.data.prim_hit = d_hit;
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));
                sbt.hitgroupRecordBase = hitgroup_record;
                sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
                sbt.hitgroupRecordCount = 1; 

                params.handle = gas_handle;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));
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

    void setVisibilityMask (int visibilityMask) {
        // set Params visibilityMask = 1, i.e. true triangles
        params.visibilityMask = visibilityMask ;
    }

    void setRayOrigin(float3* ray_origin, int size) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ray_origin), ray_origin, sizeof(float3) * size, cudaMemcpyHostToDevice));
    }

    OptixPipeline& getOptixPipeline() {
        return pipeline;
    }

    CUdeviceptr getDparams() {
        return d_params;
    }

    OptixShaderBindingTable* getSBT() {
        return &sbt;
    }

    unsigned int* getPrimitiveHit() {
        return d_hit;
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