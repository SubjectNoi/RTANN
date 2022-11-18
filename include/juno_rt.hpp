#pragma once
#ifndef JUNO_RT_H_
#define JUNO_RT_H_

#include "utils.hpp"

struct Params {
    OptixTraversableHandle handle;
};

struct RayGenData {
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
    float3* ray_origin;
    unsigned int* ray_hit;
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
    OptixAccelBuildOptions          accel_options       = {};

    OptixPipeline                   pipeline            = nullptr;
    OptixShaderBindingTable         sbt                 = {};
    char                            log[2048];
public:
    juno_rt() {
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    void constructBVHforLabelWithRadius(int _label, T** _search_points, int* _search_points_labels, int _N, int _D, T** _stat, T _radius, METRIC _metric) {
        float3* vertices;
        int num_vertices;
        std::vector <int> point_index_of_label;
        point_index_of_label.clear();
        for (int i = 0; i < _N; i++) {
            if (_search_points_labels[i] == _label) {
                point_index_of_label.push_back(i);
            }
        }
        int M;
        switch (_metric) {
            case METRIC_L2: {
                M = 2;
                int dim_pair = _D / M, hitable_num = point_index_of_label.size();
                num_vertices = hitable_num * TRIANGLE_PER_HITABLE * dim_pair;
                vertices = new float3[num_vertices];
                for (int d = 0; d < dim_pair; d++) {
                    T a = _stat[d*2][1], b = _stat[d*2+1][1];
                    T factor = sqrt(a * a + b * b);
                    // float factor = std::sqrt(_stat[d << 1][1] * _stat[d << 1][1] + _stat[d << 1 + 1][1] * _stat[d << 1 + 1][1]);
                    _radius *= factor;
                    for (int n = 0; n < hitable_num; n++) {
                        T x = _search_points[point_index_of_label[n]][d << 1];
                        T y = _search_points[point_index_of_label[n]][d << 1 + 1];
                        float3 v1 = make_float3(x - _radius, y - _radius, 1.0 * d + 1);
                        float3 v2 = make_float3(x - _radius, y + _radius, 1.0 * d + 1);
                        float3 v3 = make_float3(x + _radius, y + _radius, 1.0 * d + 1);
                        float3 v4 = make_float3(x - _radius, y - _radius, 1.0 * d + 1);
                        float3 v5 = make_float3(x + _radius, y + _radius, 1.0 * d + 1);
                        float3 v6 = make_float3(x + _radius, y - _radius, 1.0 * d + 1);
                        int base_idx = d * hitable_num * TRIANGLE_PER_HITABLE + n * TRIANGLE_PER_HITABLE;
                        vertices[base_idx + 0] = v1;
                        vertices[base_idx + 0] = v2;
                        vertices[base_idx + 0] = v3;
                        vertices[base_idx + 0] = v4;
                        vertices[base_idx + 0] = v5;
                        vertices[base_idx + 0] = v6;
                    }
                }
                CUdeviceptr d_vertices = 0;
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

                OptixAccelBufferSizes gas_buffer_sizes;
                OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));
                CUdeviceptr d_temp_buffer_gas;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
                OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
                
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
                pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
                
                std::string input;
                std::ifstream file("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/src/juno_rt.optixir", std::ios::binary);
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
                                                    1
                                                    ));

                // RayGen Sbt should be binded RUNTIME

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

                unsigned int *d_hit, *h_hit;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit), sizeof(unsigned int) * hitable_num * dim_pair));
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

    OptixPipeline& getOptixPipeline() {
        return pipeline;
    }
}; // class juno_rt

}; // namespace juno

#endif