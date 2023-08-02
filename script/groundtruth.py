import os
import sys
import time
import numpy as np
import pdb

import faiss
from datasets import load_deep1B, evaluate


# print("load data")

xb, xq, xt, gt = load_deep1B()
print (xb.shape)
print (xq.shape)
nq, d = xq.shape
# we need only a StandardGpuResources per GPU
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2 (d)
# index = faiss.index_factory(d, "IVF10000,PQ48")

# faster, uses more memory
# index = faiss.index_factory(d, "IVF4096,Flat")

co = faiss.GpuClonerOptions()

# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
co.useFloat16 = True

index = faiss.index_cpu_to_gpu(res, 0, index, co)

index.add(xb)

D, I = index.search (xq, 100)
print (I.shape)
print (I)
np.save ("/home/wtni/RTANN/RTANN/data/Deep1B/groundtruth.npy", I)