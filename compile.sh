# Copy this 3 lines to your terminal! I failed to set these envs in this file, I do not know why
# export OptiX_INSTALL_DIR=/home/altairliu/Downloads/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64 &&
# export JUNO_ROOT=/home/altairliu/workspace/RTANN &&
# export OpenBLAS_INSTALL_DIR=/home/altairliu/workspace/OpenBLAS
echo [OpeiX_INSTALL_DIR]=${OptiX_INSTALL_DIR}
echo [JUNO_ROOT        ]=${JUNO_ROOT}
cd ${JUNO_ROOT}/src
nvcc -o juno_rt.optixir --optix-ir -I ${OptiX_INSTALL_DIR}/include/ -I ${JUNO_ROOT}/include/ -I ${OpenBLAS_INSTALL_DIR}/ -I ${OptiX_INSTALL_DIR}/SDK/ -I ${OptiX_INSTALL_DIR}/SDK/support -I ${OptiX_INSTALL_DIR}/SDK/build juno_rt.cu
cd ${JUNO_ROOT}/build
cmake ..
make