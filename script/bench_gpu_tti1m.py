import os
import sys
import time
import numpy as np
import pdb
import matplotlib.pyplot as plt
import faiss
from faiss.contrib import inspect_tools
# from faiss.contrib import clustering
from datasets import load_tti1M, evaluate 
def IP(x, y):
    res = 0.0
    for i in range(len(x)):
        res += x[i] * y[i]
    return res
xb, xq, xt, gt = load_tti1M()
'''
fxb = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/search_points", "w")
for x in xb:
    for val in x:
        fxb.write("%f " % (val))
    fxb.write("\n")
fxb.close()
fxq = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/queries", "w")
for x in xq:
    for val in x:
        fxq.write("%f " % (val))
    fxq.write("\n")
fxq.close()
xb_with_id = []
nq, d = xq.shape
res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()
'''
nq, d = xq.shape
# Get 1st cluster
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, 1000, faiss.METRIC_INNER_PRODUCT)

index.train(xb)

# get 1st cluster centroids
centroids = index.quantizer.reconstruct_n(0, index.nlist)
centroids_with_id = []
for i in range(len(centroids)):
    centroids_with_id.append([i, centroids[i]])
print(centroids.shape)
# train a flat index to get label (centroids is search point, search top-1 -> cluster centroids id)
get_label = faiss.IndexFlatIP(d)
get_label.add(centroids)
D, I = get_label.search(xb, 1)
'''
for i in range(100):
    query = xb[i]
    centroids_with_id.sort(key=lambda x : IP(query, x[1]))
    centroids_with_id = centroids_with_id[::-1]
    print(I[i], centroids_with_id[0][0])
'''
labels = I.reshape(-1)
print(labels.shape)
nlists = 1000
f1 = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/parameter_0/cluster_centroids_%d" % (nlists), "w")
for cc in centroids:
    for x in cc:
        f1.write("%f " % (x))
    f1.write("\n")
f1.write("-----\n")
for l in labels:
    f1.write("%d " % (l))
f1.close()

cluster_mapping = {}
for i in range(len(labels)):
    l = labels[i]
    if l not in cluster_mapping:
        cluster_mapping[l] = []
    cluster_mapping[l].append(i)
cnt = 1
# 1000 clusters
for keys in range(1000):
    xb_of_this_cluster = [xb[i] for i in cluster_mapping[keys]]
    # 100 subspaces
    for d in range(100):
        X = [[x[2*d], x[2*d+1]] for x in xb_of_this_cluster]
        X = np.array(X)
        # train cluster with inner product
        sub_quantizer = faiss.IndexFlatIP(2)
        sub_index = faiss.IndexIVFFlat(sub_quantizer, 2, 32, faiss.METRIC_INNER_PRODUCT)
        sub_index.train(X)
        sub_centroids = sub_index.quantizer.reconstruct_n(0, sub_index.nlist)
        
        # get label
        get_sub_label = faiss.IndexFlatIP(2)
        get_sub_label.add(sub_centroids)
        SD, SI = get_sub_label.search(X, 1)
        sub_labels = SI.reshape(-1)
        
        fcluster = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/parameter_0/codebook_%d/codebook_cluster=%d_dim=%d" % (nlists, keys, d), "w")
        for item in sub_centroids:
            for x in item:
                fcluster.write("%f " % (x))
            fcluster.write("\n")
        fcluster.write("-----\n")
        for item in sub_labels:
            fcluster.write("%d " % (item))
        fcluster.close()
        print("%8d / %8d, %8d / %8d" % (keys, nlists, d, 100), end='\r')
        
        del get_sub_label
        del sub_quantizer
        del sub_index

'''
from numba import njit, prange
@numba.jit(nopython=True, parallel=True)
def get_labels(xb, centroids_with_id):
    labels = np.zeros((1000000))
    for i in prange(1000000):
        query = xb[i]
        centroids_with_id.sort(key=lambda x : IP(query, x[1]))
        centroids_with_id = centroids_with_id[::-1]
        # labels.append(centroids_with_id[0][0])
        # print("%d" %(i), end='\r')
        labels[i] = centroids_with_id[0][0]
    return labels

l = get_labels(xb, centroids_with_id)
print(l.shape)
''' 
'''
index = faiss.index_factory(d, "IVF4096,PQ5", faiss.METRIC_INNER_PRODUCT)
co = faiss.GpuClonerOptions()
co.useFloat16 = True

index = faiss.index_cpu_to_gpu(res, 0, index, co)
print("train")

index.train(xb)

# print("add vectors to index")

index.add(xb)

print("benchmark")
lim = int(sys.argv[1])
for lnprobe in range(lim, lim+1):
    nprobe = 1 << lnprobe
    index.setNumProbes(nprobe)
    t, r = evaluate(index, xq, gt, 100)

    print("QPS: % 10.3f r1@100: % 7.4f" % ((10000.0) / (t / 1000.0), r[100]))
'''
