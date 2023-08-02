import os
import time
import numpy as np
import pdb
import sys
import faiss
from faiss.contrib import inspect_tools
# from faiss.contrib import clustering
from datasets import load_sift1B, evaluate, load_sift1M
xb, xq, xt, gt = load_sift1B()
nq, d = xq.shape
res = faiss.StandardGpuResources()
xt = xb[0:10000000]
index = faiss.index_factory(d, "IVF10000,Flat")
# index = faiss.index_factory(d, "IVF10000,PQ64")
co = faiss.GpuClonerOptions()
co.useFloat16 = True

index = faiss.index_cpu_to_gpu(res, 0, index, co)

index.train(xt)
print("Training Finished")
index.add(xb)
'''
# ivf = faiss.contrib.inspect_tools.get_invlist_sizes(index.invlists)
centroids = index.quantizer.reconstruct_n(0, index.nlist)
print(centroids.shape)
get_label = faiss.IndexFlatL2(d)
get_label.add(centroids)
label = get_label.search(x=xb, k=1)[1].reshape(-1)
print(len(label))
f = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1B/parameter_1/cluster_centroids_10000", "w")
for cc in centroids:
    for x in cc:
        f.write("%f " % (x))
    f.write("\n")
f.write("-----\n")
for l in label:
    f.write("%d " % (l))
f.close()
print("Cluster Writing Finished")
# print("add vectors to index")

index.add(xb)
xb_encoded = index.pq.compute_codes(xb)
codebook = faiss.contrib.inspect_tools.get_pq_centroids(index.pq)
f = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1B/parameter_1/codebook", "w")
for d in range(64):
    for c in range(256):
        f.write("%f %f\n" % (codebook[d][c][0], codebook[d][c][1]))
    f.write("\n\n")
f.close()
cluster_mapping = {}
for _l in range(len(label)):
    l = label[_l]
    if l not in cluster_mapping:
        cluster_mapping[l] = []
    cluster_mapping[l].append(_l)
for key in range(10000):
    xbofkey_encoded = [xb_encoded[i] for i in cluster_mapping[key]]
    f = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1B/parameter_1/codebook_10000/points_in_cluster_%d_label" % (key), "w")
    for d in range(64):
        for item in xbofkey_encoded:
            f.write("%d " % (item[d]))
        f.write("\n")
    f.close()
    print("%6d / %6d" % (key, 10000), end='\r')
'''

for lnprobe in range(10):
    _nprobe = 1 << lnprobe
    index.nprobe = _nprobe
    t, r = evaluate(index, xq, gt, 100)
    
    print("QPS: % 10.3f r1@100: % 7.4f" % ((10000.0) / (t / 1000.0), r[100]))

