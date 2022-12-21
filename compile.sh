cd /home/wtni/RTANN/RTANN/src
nvcc -o juno_rt.optixir --optix-ir -I /home/wtni/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/include/ -I /home/wtni/RTANN/RTANN/include/ -I /home/wtni/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/SDK/ -I /home/wtni/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/SDK/support -I /home/wtni/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/SDK/build juno_rt.cu
cd /home/wtni/RTANN/RTANN/build
cmake ..
make