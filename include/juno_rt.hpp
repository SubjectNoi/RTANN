#pragma once
#ifndef JUNO_RT_H_
#define JUNO_RT_H_

#include "utils.hpp"

namespace juno {

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

public:
    juno_rt() {
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    void constructBVHforLabelWithRadius(int _label, T** _search_points, int* _search_points_labels, int _N, int _D, T _radius, METRIC _metric) {
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