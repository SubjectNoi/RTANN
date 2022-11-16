#include "include/rtann.h"
#include "include/dbg.h"
#include <iostream>
#include <assert.h>
#include <math.h>
template <typename T>
T* allocate(int d0) {
    T* res = new T[d0];
    return res;
}


template <typename T>
T** allocate(int d0, int d1) {
    T** res = new T*[d0];
    for (int i = 0; i < d0; i++) {
        res[i] = new T[d1];
    }
    return res;
}

template <typename T>
T*** allocate(int d0, int d1, int d2) {
    T*** res = new T**[d0];
    for (int i = 0; i < d0; i++) {
        res[i] = new T*[d1];
        for (int j = 0; j < d1; j++) {
            res[i][j] = new T[d2];
        }
    }
    return res;
}

int main(int argc, char** argv) {
    int nq = atoi(argv[1]), d = atoi(argv[2]), m = d / 2, nbits = atoi(argv[3]), n_points = 1 << nbits;
    float dist_thres_scale = atof(argv[4]);
    float* codebook_dist_med = allocate<float>(m);
    float** queries = allocate<float>(nq, d);
    float*** codebook = allocate<float>(m, n_points, 2);
    float* dummy3;
    float** dummy4;
    std::vector<std::vector<float>> result;
    dbg("Loading Query... ");
    rtann::load_query("/home/wtni/RTANN/RTANN/data/query_toy.txt", nq, d, queries);
    dbg("Loading Codebook... ");
    rtann::load_codebook("/home/wtni/RTANN/RTANN/data/codebook_toy.txt", m, nbits, codebook, codebook_dist_med);
    dbg("Load Finished");
    rtann::search(queries, nq, d, codebook, m, nbits, codebook_dist_med, dist_thres_scale, dummy3, dummy4, result);
    dbg("OptiX-based Search Complete");
    return 0;
}
