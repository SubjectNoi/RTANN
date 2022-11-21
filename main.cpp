#include "juno.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string path = "/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1M/";
    juno::juno_core<float> server(path, SIFT1M);
    juno::juno_query_total<float> query(path, SIFT1M);
    query.generateQueryBatch(100);
    // server.setupBVHDict();
    server.serveQueryBatch(query.query_queue[19]);
    return 0;
}
