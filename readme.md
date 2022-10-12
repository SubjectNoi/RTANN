# Interface Definition

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
