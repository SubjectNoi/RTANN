#include "juno.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // server.plotDataset(&query);
    // query.generateQueryBatch(1);
    // server.setupBVHDict();
    // server.serveQueryBatch(query.query_queue[0]);

    // server.buildJunoIndex();
    // query.generateQueryBatch(10000);
    // server.serveQuery(query.query_queue[0], 8);

    // std::string path = std::getenv("JUNO_ROOT") + std::string("/data/SIFT1M/");
    std::string path = "/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/" ;
    juno::juno_core<float> server(path, TTI1M, 1000);
    juno::juno_query_total<float> query(path, TTI1M);
    query.generateQueryBatch(10000);
    int nlists;
    float radius ;
    while (std::cin >> nlists >> radius) {
        if (nlists < 0) break;
        server.buildJunoIndexWhole(radius);
        server.serveQueryWhole(query.query_queue[0], nlists);
    }


    // std::string path = "/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/dummy/";
    // juno::juno_core<float> server(path, CUSTOM, 1, 0.3);
    // juno::juno_query_total<float> query(path, CUSTOM);
    // server.buildJunoIndexWhole();
    // query.generateQueryBatch(1);
    // server.serveQueryWhole(query.query_queue[0], 1);
    return 0;
}
