#pragma once
#ifndef JUNO_QUERY_H_
#define JUNO_QUERY_H_

#include "utils.hpp"

namespace juno {

template <typename T>
class juno_query {
private:
    int                 query_id;
    int                 D;
    T*                  data;
    int                 reference;
    int                 result;
public:
    juno_query() {}
}; // class juno_query

template <typename T>
class juno_query_batch {
private:
    int                             query_batch_id;
    int                             size;
    std::vector<juno_query<T>>      queries;
public:
    juno_query_batch() {}
}; // class juno_query_batch

}; // namespace juno

#endif