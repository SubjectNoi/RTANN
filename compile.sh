cd /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/src
nvcc -o juno_rt.optixir --optix-ir -I /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/include/ -I /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/include/ -I /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/SDK/ -I /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/SDK/support -I /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/SDK/build juno_rt.cu
cd /home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/build
cmake ..
make