cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(cmake/CPM.cmake)
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

project(RTANN LANGUAGES C CXX CUDA)
# find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -mavx2 -pg -w")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86;)
set(OptiX_INSTALL_DIR /home/wtni/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/)
set(OpenBLAS_INSTALL_DIR /home/wtni/OpenBLAS/build)
set(CuCollection_INSTALL_DIR /home/wtni/cuCollections/)

include_directories(${OptiX_INSTALL_DIR}/include)
include_directories(${OptiX_INSTALL_DIR}/SDK)
include_directories(${OptiX_INSTALL_DIR}/SDK/support)
include_directories(${OptiX_INSTALL_DIR}/SDKBuild)
include_directories(/home/wtni/RTANN/RTANN/include)
include_directories(${OpenBLAS_INSTALL_DIR}/include)
include_directories(/usr/local/cuda/samples/common/inc/)
# include_directories(/usr/local/cuda/include/thrust/)
# include_directories(/usr/local/cuda/include/)
include_directories(/usr/local/cuda/include/lib64/)
include_directories(${CuCollection_INSTALL_DIR}/include/)
link_directories(/usr/lib /usr/local/lib /usr/local/cuda/lib64)
link_directories(${OpenBLAS_INSTALL_DIR}/lib)

# cuda_add_executable(main main.cpp src/juno_gpu_kernel.cu OPTIONS -arch=sm_86)
# set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_ARCHITECTURES 86)
add_executable(main main.cpp src/juno_gpu_kernel.cu)

# target_include_directories(main PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# target_compile_options(main PRIVATE --compiler-options=-Wall --compiler-options=-Wextra --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)

target_link_libraries(main ${CUDA_cudadevrt_LIBRARY} openblas OpenMP::OpenMP_CXX cuco)