This file will be used as developer log from now on.

# Planning of Re-constructing the project
1. Expected usage of our system:
```
int main(int argc, char** argv) {
    RTANN rtann_instance;
    // Offline part
    rtann_instance.read_data(file_path);
    rtann_instance.set_property("coarse_grained_cluster_num", int32);
    rtann_instance.set_property("use_pq", true/false);
    rtann_instance.set_property("rt_mode", query_as_ray/points_as_ray);

    rtann_instance.coarse_grained_cluster();
    if (query_as_ray) rtann_instance.build_bvh();
    // Online part
    vector <query> q;   // Other threads may fill the q
    for (int i = 0; i < q.size(); i++) {
        // No pipeline version:
        lab = rtann_instance.search_coarse_grain_cluster(q[i]);
        bvh = rtann_instance.bvh_dict[lab];
        pts = rtann_instance.points[label==lab];

        if (query_as_ray) {
            rtann_instance.launch_RT(primitives=bvh, ray=q[i]);
        }
        else {
            bvh_query = rtann_instance.build_bvh_query(pts);
            rtann_instance.launch_RT(primitives=bvh_query, ray=pts);
        }

        rtann_instance.count_intersection();
        rtann_instance.return_top100();
    }
    return 0;
}
```

# Interface Definition of count_intersect

1. Total points = `N`, Select points = `C`. For every `N` elements of `points` (representing a single query), you need to calculate the occurance of a point in `C` in `D/2` iteration. Example:
* `N = 16, C = 4, D = 4, Q = 2`
```
points = {
/* 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 */    
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    // Query 1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    // Query 2
};

candidate = {
    /*<-C(0~1)->*/   /*<-C(2~3)->*/
    2, 15,  7,  9,   15, 7,  1, 13,          // Query 1
    /*<-C(0~1)->*/   /*<-C(2~3)->*/
    4,  5, 11,  0,   12, 9,  4,  5,          // Query 2
};

accu<<<?, ?>>>(points, candidate);

// points should be:

points = {
/* 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 */ 
    0, 1, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2,    // Query 1
    1, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,    // Query 2
};
```
```
void __global__ accu(int*   points,         /* N * Q */
               const int*   candidate,      /* C * D/2 * Q */);

// Hint: Let every thread block process a query, N=8192, C=2048, D=128, now processing query 3, which is:
//          points[3 * 8192      : 3 * 8192      + 8192     ]
//       candidate[3 * 2048 * 64 : 3 * 2048 * 64 + 2048 * 64]
// Given we have 1024 threads per block, you need to:
// First iterate over 64 2-dim pairs (Don't map this iterator to 1024 threads, we call it Temporal Mapping)
// Second iterate 1024 threads over 2048 candidate elements (Map this iterator to 1024 threads, we call it Spatial Mapping)
//
// for (iter : 0 to 64)
//     for (thread_id : 0 to 2048 step 1024)
//         points[candidate[thread_id]]++      
//
// Of course the array index should be adjusted by query for correct result, where query_id = block_id.
// If you have learnt the usage of __shared__, you can put points (length=8192) to shared memory (32kB), you can also put 1/8 of candidate to shared memory (64kB) 
