cd /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/src
nvcc -o rtann.optixir --optix-ir -I /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/include/ -I /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/include/ rtann.cu
cd /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/build
cmake ..
make