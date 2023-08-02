import os
import sys
import time
import numpy as np
import pdb

import faiss
from datasets import load_deep1B, evaluate


# print("load data")

xb, xq, xt, gt = load_deep1B()
nq, d = xq.shape
# we need only a StandardGpuResources per GPU
res = faiss.StandardGpuResources()
index = faiss.index_factory(d, "IVF20000_HNSW32,PQ12")

# faster, uses more memory
# index = faiss.index_factory(d, "IVF4096,Flat")

co = faiss.GpuClonerOptions()

# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
co.useFloat16 = True

index = faiss.index_cpu_to_gpu(res, 0, index, co)

# print("train")

index.train(xt)

# print("add vectors to index")

index.add(xb)

for lnprobe in range(10):
    _nprobe = 1 << lnprobe
    index.nprobe = _nprobe
    t, r = evaluate(index, xq, gt, 100)
    
    print("QPS: % 10.3f r1@100: % 7.4f" % ((10000.0) / (t / 1000.0), r[100]))


# print("benchmark")
# lnprobe = int(sys.argv[1])
# nprobe = 1 << lnprobe
# nprobe = int(sys.argv[1])
# index.setNumProbes(nprobe)
# t, r = evaluate(index, xq, gt, 100)

# print("QPS: % 10.3f r1@100: % 7.4f" % ((10000.0) / (t / 1000.0), r[100]))
# print("Latency: %10.3f (us)" % (t))
