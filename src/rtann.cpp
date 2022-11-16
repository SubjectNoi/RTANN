#include "rtann.h"
#include "dbg.h"
#include <iostream>
#include <random>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
// #include <optix.h>
// #include <optix_function_table_definition.h>
// #include <optix_stack_size.h>
// #include <optix_stubs.h>
#include <sys/resource.h>

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <algorithm>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#define DUMMY_INPUT_FOR_TEST 0      // Set to one if you don't want to run real data
                                    // This is useful when you want to try different 
                                    // parameter but have no real data
template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 0.0f, -1.0f} );
    cam.setLookat( {0.0f, 0.0f, 8192.0f} );
    cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setFovY( 60.0f );
    cam.setAspectRatio( (float)width / (float)height );
}

namespace rtann {

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

void load_query(const char* query_path, const int& NQ, const int& D, float** queries) {
#if DUMMY_INPUT_FOR_TEST == 1
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<> uniform(-0.5f, 0.5f);
    for (int q = 0; q < NQ; q++) {
        for (int d = 0; d < D; d++) {
            queries[q][d] = uniform(engine);
        }
    }
#else
    std::ifstream fread_query(query_path, std::ios::in);
    for (int q = 0; q < NQ; q++) {
        for (int d = 0; d < D; d++) {
            fread_query >> queries[q][d];
        }
    }
    fread_query.close();
#endif
}

void load_codebook(const char* codebook_path, const int& M, const int& nbits, float*** codebook, float* dist_medium) {
    int N_POINTS = 1 << nbits;
#if DUMMY_INPUT_FOR_TEST == 1
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<> uniform(-0.5f, 0.5f);
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < N_POINTS; i++) {
            codebook[m][i][0] = uniform(engine);
            codebook[m][i][1] = uniform(engine);
        }
        float tmp_res = 0.0f;
        int tmp_cnt = 0;
        for (int x = 0; x < N_POINTS; x++) {
            for (int y = 0; y < N_POINTS; y++) {
                if (x == y) continue;
                else {
                    float a = codebook[m][x][0] - codebook[m][y][0];
                    float b = codebook[m][x][1] - codebook[m][y][1];
                    tmp_res += sqrt(a * a + b * b);
                    tmp_cnt++;
                }
            }
        }
        dist_medium[m] = tmp_res / (1.0 * tmp_cnt);
        dist_medium[m] /= 1;
    }
#else
    std::ifstream fread_codebook(codebook_path, std::ios::in);
    std::vector <float> vd;
    vd.clear();
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < N_POINTS; i++) {
            fread_codebook >> codebook[m][i][0] >> codebook[m][i][1];
        }
        float tmp_res = 0.0f;
        int tmp_cnt = 0;
        for (int x = 0; x < N_POINTS; x++) {
            for (int y = 0; y < N_POINTS; y++) {
                if (x == y) continue;
                else {
                    float a = codebook[m][x][0] - codebook[m][y][0];
                    float b = codebook[m][x][1] - codebook[m][y][1];
                    vd.push_back(sqrt(a * a + b * b));
                    tmp_res += sqrt(a * a + b * b);
                    tmp_cnt++;
                }
            }
        }
        dist_medium[m] = tmp_res / (1.0 * tmp_cnt);
    }
    fread_codebook.close();
#endif
}

void search(      float**   queries,            /* NQ * D */
            const int&      NQ, 
            const int&      D,
                  float***  codebook,           /* M * 2^nbits * 2 */
            const int&      M,                  /* D / 2 == M */
            const int&      nbits, 
            const float*    dist_thres,
            const float     dist_thres_scale,
            // const bool&     search_as_primitives,    // This one is changed to macro and define pre-compile
                  float*    distance,
                  float**   index,
                  std::vector<std::vector<float>>& res) 
{
    // Initialize OptiX
    assert((D / 2 == M) && "[Error] Dimension and codebook number mismatch!");
    char log[2048];
    OptixDeviceContext context = nullptr;
    CUDA_CHECK(cudaFree(0));
    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const int N_POINTS = (1 << nbits);
    int num_vertices = -1;
    int num_ray = -1;
    int num_primitives = -1;
    float3* vertices;
    float3* ray_origin;
#if SEARCH_AS_PRIMITIVES == 0
    // Search as ray, query as primitive
    num_ray = N_POINTS * M;
    num_primitives = NQ * M;
    num_vertices = 2 * 3 * num_primitives; 
    /* Since all object in CG are represented with triangle, rather than create a hittable ball.
     * I choose to create a hittable square with only 2 triangles (ball need a lot of triangles)
     * You can treat the new hittable objects applying inf-norm rather than 2-norm.
     */
    dbg("Search => Ray, Query => Primitives", num_ray, num_primitives);
    vertices = new float3[num_vertices];
    ray_origin = new float3[num_ray];

    // Create Hitable Primitives
    for (int d = 0; d < D; d+=2) {
        for (int q = 0; q < NQ; q++) {
            float x = 1.0 * queries[q][d], y = 1.0 * queries[q][d + 1];
            float _dist_thres;
#if USE_EMPRICAL_THRES == 0            
            _dist_thres = dist_thres_scale * dist_thres[d / 2];
#else
            _dist_thres = dist_thres_scale;
#endif
            float3 v1 = make_float3(x - _dist_thres, y - _dist_thres, 1.0 * d / 2 + 1);
            float3 v2 = make_float3(x - _dist_thres, y + _dist_thres, 1.0 * d / 2 + 1);
            float3 v3 = make_float3(x + _dist_thres, y + _dist_thres, 1.0 * d / 2 + 1);
            float3 v4 = make_float3(x - _dist_thres, y - _dist_thres, 1.0 * d / 2 + 1);
            float3 v5 = make_float3(x + _dist_thres, y + _dist_thres, 1.0 * d / 2 + 1);
            float3 v6 = make_float3(x + _dist_thres, y - _dist_thres, 1.0 * d / 2 + 1);
            int base_idx = (d / 2) * NQ * 6 + q * 6;
            vertices[base_idx + 0] = v1;
            vertices[base_idx + 1] = v2;
            vertices[base_idx + 2] = v3;
            vertices[base_idx + 3] = v4;
            vertices[base_idx + 4] = v5;
            vertices[base_idx + 5] = v6;
        }
    }

        // Create Rays
    for (int m = 0; m < M; m++) {
        for (int p = 0; p < N_POINTS; p++) {
            float x = 1.0 * codebook[m][p][0], y = 1.0 * codebook[m][p][1];
            ray_origin[m * N_POINTS + p] = make_float3(x, y, 1.0 * m + 0.5); 
        }
    }
#else
    // Search as primitive, query as ray
    num_ray = NQ * M;
    num_primitives = N_POINTS * M;
    num_vertices = 2 * 3 * num_primitives;
    dbg("Search => Primitives, Query => Ray", num_primitives, num_ray);
    vertices = new float3[num_vertices];
    ray_origin = new float3[num_ray];

    // Create Hitable Primitives
    for (int m = 0; m < M; m++) {
        for (int p = 0; p < N_POINTS; p++) {
            float x = 1.0 * codebook[m][p][0], y = 1.0 * codebook[m][p][1];
            float _dist_thres;
#if USE_EMPRICAL_THRES == 0            
            _dist_thres = dist_thres_scale * dist_thres[m];
#else
            _dist_thres = dist_thres_scale;
#endif
            float3 v1 = make_float3(x - _dist_thres, y - _dist_thres, 1.0 * m + 1);
            float3 v2 = make_float3(x - _dist_thres, y + _dist_thres, 1.0 * m + 1);
            float3 v3 = make_float3(x + _dist_thres, y + _dist_thres, 1.0 * m + 1);
            float3 v4 = make_float3(x - _dist_thres, y - _dist_thres, 1.0 * m + 1);
            float3 v5 = make_float3(x + _dist_thres, y + _dist_thres, 1.0 * m + 1);
            float3 v6 = make_float3(x + _dist_thres, y - _dist_thres, 1.0 * m + 1);
            int base_idx = m * N_POINTS * 6 + p * 6;
            vertices[base_idx + 0] = v1;
            vertices[base_idx + 1] = v2;
            vertices[base_idx + 2] = v3;
            vertices[base_idx + 3] = v4;
            vertices[base_idx + 4] = v5;
            vertices[base_idx + 5] = v6;
        }
    }

    // Create Rays
    for (int d = 0; d < D; d+=2) {
        for (int q = 0; q < NQ; q++) {
            float x = 1.0 * queries[q][d], y = 1.0 * queries[q][d + 1];
            ray_origin[(d / 2) * NQ + q] = make_float3(x, y, 1.0 * (d / 2) + 0.5); 
        }
    }
#endif
    float3* d_ray_origin;
    const size_t ray_origin_size = sizeof(float3) * num_ray;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin), ray_origin_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ray_origin), ray_origin, ray_origin_size, cudaMemcpyHostToDevice));
#if SEARCH_AS_PRIMITIVES == 0
    // If we treat codebook point as ray, query as primitives, to get the codebook points 
    // with in the distance threshold between query, we just need to return the ray id. 
    unsigned int* d_ray_hit;
    const size_t ray_hit_size = sizeof(unsigned) * num_ray;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_hit), ray_hit_size));
#else
    // Or,
    // we need to return the id of the primitives a ray hit.
    // So, I still have bug here, under this implementation,
    // I can only get one primitive ID, although there are  
    // multiple primitives a ray hit in total. 
    unsigned int* d_prim_hit;
    const size_t prim_hit_size = sizeof(unsigned) * num_primitives; // Every prim have 2 triangles
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prim_hit), prim_hit_size));
#endif

    // Dump the primitives into the OptiX Scene
    const size_t vertices_size = sizeof(float3) * num_vertices;
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices, vertices_size, cudaMemcpyHostToDevice));

    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(num_vertices);
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // Allocate OptiX space
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
    OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));

    // Configure the OptiX module
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

    // Load the shader binary and create OptiX module
    // The compiling bash will first call nvcc to compile .cu file into optixir file
    // Then the optixir file are load here, so g++ should be invoke AFTER nvcc
    std::string input;
    std::ifstream file("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/src/rtann.optixir", std::ios::binary);
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

    // Binding shader, the shader will be CALLBACK under specific circumstance
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    // Ray-Generation, will be called once the OptiX pipeline is launched
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
    
    // Miss, will be called if a ray hit nothing
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

    // Hit Group, any hit will be called once a ray hit something, 
    // closest hit will be called at last to return the closest
    // object a ray hit (for depth/z-buffer check usage)
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

    // Create the pipeline
    OptixPipeline pipeline = nullptr;
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
    // Configure the shader binding table
    OptixShaderBindingTable sbt = {};
    CUdeviceptr raygen_record;      
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
    sutil::Camera cam;
    configureCamera(cam, 100, 100);
    RayGenSbtRecord rg_sbt;
    rg_sbt.data = {};
    rg_sbt.data.cam_eye = cam.eye();
    rg_sbt.data.ray_origin = d_ray_origin;      // IMPORTANT: Pass the data from host to device.
#if SEARCH_AS_PRIMITIVES == 0
    rg_sbt.data.ray_hit = d_ray_hit;            // Device will write this list to record ray hit
#else
    rg_sbt.data.prim_hit = d_prim_hit;
#endif
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

    unsigned int* d_phit, *h_phit;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_phit), sizeof(unsigned int) * 16));
    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    hg_sbt.data.prim_hit = d_phit;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;  

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Params params;
    params.handle = gas_handle;
    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(Params), cudaMemcpyHostToDevice));
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st);
    // Ray Tracing Begin
    OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, num_ray, 1, 1));
    cudaEventRecord(ed);
    CUDA_SYNC_CHECK();
    unsigned int* ray_hit;
    unsigned int* prim_hit;
#if SEARCH_AS_PRIMITIVES == 0
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ray_hit), ray_hit_size));
    CUDA_CHECK(cudaMemcpy(ray_hit, reinterpret_cast<void*>(d_ray_hit), ray_hit_size, cudaMemcpyDeviceToHost));
#else
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&prim_hit), prim_hit_size));
    CUDA_CHECK(cudaMemcpy(prim_hit, reinterpret_cast<void*>(d_prim_hit), prim_hit_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_phit), sizeof(unsigned int) * 16));
    for (int i = 0; i < 16; i++) h_phit[i] = 0;
    CUDA_CHECK(cudaMemcpy(h_phit, reinterpret_cast<void*>(d_phit), sizeof(unsigned int) * 16, cudaMemcpyDeviceToHost));
#endif
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "OptiX Trace Time: " << ms << " ms" << std::endl;
    std::vector <std::pair<int, int>> hit_codebook_entry;

    /* Here you can use some toy example for comprehending:
     * How to get the codebook ID that are hit by queries
     * You can load the data: data/codebook_toy.txt, data/query_toy.txt
     * And draw the triangle and ray on a paper to see which codebook
     * entry should be hit (Setting threshold to 0.1 and nbits to 2)
     */
#if SEARCH_AS_PRIMITIVES == 0
    for (int i = 0; i < num_ray; i++) {
        if (ray_hit[i] == MAGIC_NUMBER_0) {
            hit_codebook_entry.push_back(std::pair<int, int>(i / N_POINTS, i % N_POINTS));
        }
    }
    for (auto&& p : hit_codebook_entry) std::cout << p.first << " " << p.second << std::endl;
#else
    for (int i = 0; i < num_primitives * 2; i++) {
        if (prim_hit[i] == MAGIC_NUMBER_1) {
            hit_codebook_entry.push_back(std::pair<int, int>(i / (N_POINTS), (i) % N_POINTS));
        }
    }    
    for (auto&& p : hit_codebook_entry) std::cout << p.first << " " << p.second << std::endl;
    std::cout << "------------------------" << std::endl;
    for (int i = 0; i < 16; i++) {
        if (h_phit[i] == 114514) {
            std::cout << i << std::endl;
        }
    }
#endif

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

    OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( module ) );
    OPTIX_CHECK( optixModuleDestroy( triangle_module ) );

    OPTIX_CHECK( optixDeviceContextDestroy( context ) );                         
} // void search()

}; // namespace rtann
