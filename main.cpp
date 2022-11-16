#include "juno.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string path = "/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1M/";
    juno::juno_core<float> server(path, SIFT1M);
    server.setupBVHDict();
    return 0;
}
