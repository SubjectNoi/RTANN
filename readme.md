# Interface Definition

```
// Search a query among codebook

search(float** query /* NQ*D */, int NQ, float** codebook /* M(=D/2) * 2^nbits */, int M, int nbits, bool search_as_ray /* ==true: query-->triangle, search-->ray, ==false: query-->ray, search-->triangle */);
```


