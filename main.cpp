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

    // std::string path = std::getenv("JUNO_ROOT") + std::string("/data/DEEP1M/");
    std::string path = "/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1M/" ;
    juno::juno_core<float> server(path, SIFT1M, 1000);
    juno::juno_query_total<float> query(path, SIFT1M);
    query.generateQueryBatch(10000);
    int nlists;
    float a, b;
    // for (int nlists = 1; nlists < 2; nlists=nlists*2) {
    //     for (int al = 100; al <= 100; al++) {
    //         for (int be = 5; be <= 100; be+=5) {
    //             float alpha = (1.0 * al) / 100.0;
    //             float beta = (1.0 * be) / 100.0;
    //             server.buildJunoIndexWhole(alpha, beta);
    //             server.serveQueryWhole(query.query_queue[0], nlists);

    //         }
    //     }
    // }
                //Deep1M: 32 0.13 0.6
    while(std::cin >> nlists >> a >> b) {
                if (nlists == 114514) break;
                server.buildJunoIndexWhole(a, b);
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
