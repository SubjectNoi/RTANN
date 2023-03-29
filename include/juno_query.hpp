#pragma once
#ifndef JUNO_QUERY_H_
#define JUNO_QUERY_H_

#include "utils.hpp"

namespace juno {

template <typename T>
class juno_query_batch;
template <typename T>
class juno_query_total {
private:
    std::string     dataset_dir;

    T**             queries;
    T*              queries_flatten;
    int**           ground_truth;
    int*            ground_truth_flatten;
    int             Q;
    int             D;
    METRIC          metric;
public:
    std::vector <juno_query_batch<T>*> query_queue;
public:
    juno_query_total(std::string _dataset_dir,
               DATASET ds=CUSTOM
              ) 
    {
        dataset_dir = _dataset_dir;
        switch (ds) {
            case SIFT1M:
                Q = 10000;
                D = 128;
                metric = METRIC_L2;
                break;
            case SIFT1B:

                break;
            case TTI1M:
                Q = 10000;
                D = 200;
                metric = METRIC_MIPS;
                break;
            case TTI1B:

                break;
            case DEEP1M:
                Q = 10000;
                D = 96;
                metric = METRIC_L2;
                break;
            case CUSTOM:
                Q = 1;
                D = 4;
                metric = METRIC_L2;
                break;
            default:

                break;
        }
        dbg("Beginning Reading Query ...");
        queries = new T* [Q];
        queries_flatten = new T[Q * D];
        for (int i = 0; i < Q; i++) queries[i] = new T[D];
        read_queries((dataset_dir + "queries").c_str(), queries, Q, D);
        for (int q = 0; q < Q; q++) {
            for (int d = 0; d < D; d++) {
                queries_flatten[q * D + d] = queries[q][d];
            }
        }

        ground_truth = new int*[Q];
        ground_truth_flatten = new int[Q * 100];
        for (int i = 0; i < Q; i++) ground_truth[i] = new int[100];
        read_ground_truth((dataset_dir + "ground_truth").c_str(), ground_truth, Q);
        for (int q = 0; q < Q; q++) {
            for (int gt = 0; gt < 100; gt++) {
                ground_truth_flatten[q * 100 + gt] = ground_truth[q][gt];
            }
        }
        dbg("Finished Reading Query ...");
    }

    void generateQueryBatch(int _size) {
        for (int i = 0; i * _size < Q; i++) {
            auto query_batch = new juno_query_batch<T>(i, _size, D, queries, i * _size);
            query_queue.push_back(query_batch);
        }
    }

    T** getQueryData() {
        return queries;
    }

    T* getQueryDataFlatten() {
        return queries_flatten;
    }

    int** getGroundTruth() {
        return ground_truth;
    }

    int* getaGroundTruthFlatten() {
        return ground_truth_flatten;
    }

    int getQueryAmount() {
        return Q;
    }
}; // class juno_query

template <typename T>
class juno_query_batch {
private:
    int             query_batch_id;
    int             size;
    int             D;
    T**             query_total_data;
    int             start_idx;
    T**             data;
    T*              data_flatten;
public:
    juno_query_batch(int _query_batch_id,
                     int _size,
                     int _D,
                     T** _query_data,
                     int _start_idx
                    ) 
    {
        query_batch_id = _query_batch_id;
        size = _size;
        D = _D;
        query_total_data = _query_data;
        start_idx = _start_idx;

        data = new T*[size];
        data_flatten = new T[size * D];
        for (int s = 0; s < size; s++) data[s] = new T[D];
        for (int s = 0; s < size; s++) {
            memcpy(data[s], query_total_data[start_idx + s], sizeof(T) * D);
        }
        for (int s = 0; s < size; s++) {
            for (int d = 0; d < D; d++) {
                data_flatten[s * D + d] = data[s][d];
            }
        }
    }

    T** getQueryData() { return data; }

    T* getFlattenQueryData() { return data_flatten; }

    int getQuerySize() { return size; }
}; // class juno_query_batch

}; // namespace juno

#endif