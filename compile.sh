cd /home/wtni/RTANN/RTANN/src
nvcc -o rtann.optixir --optix-ir -I /home/wtni/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/include/ -I /home/wtni/RTANN/RTANN/include/ rtann.cu
cd /home/wtni/RTANN/RTANN/build
cmake ..
make