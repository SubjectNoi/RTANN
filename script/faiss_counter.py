import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
# from numba import njit
import numpy as np
import pdb
# import torch
import faiss
from faiss.contrib import inspect_tools
from datasets import load_sift1M, load_deep1M, load_tti1M
N = 1000000
xb, xq, xt, gt = load_tti1M()
nq, D = xq.shape
nlist = 1000
nprobe = 8
res = faiss.StandardGpuResources()

coarse_quantizer = faiss.IndexFlatL2 (D)
index = faiss.IndexIVFPQ (coarse_quantizer, D,
                          1000, 40, 8)
index.nprobe = 8
# index = faiss.index_factory(D, "IVF1000,PQ40")
co = faiss.GpuClonerOptions()
co.useFloat16 = True
M = 5
pq_d = D // M
index.train(xb)
index.add(xb)

codebook = faiss.contrib.inspect_tools.get_pq_centroids(index.pq) # codebook: 40 * 256 * 5
print(codebook.shape)
centroids = index.quantizer.reconstruct_n(0, index.nlist) # cluster centroids: nlist * d
print (centroids.shape)
get_label_index = faiss.IndexFlat(D)
get_label_index.train(centroids)
get_label_index.add(centroids)
labels = get_label_index.search(x=xb, k=1)[1].reshape(-1) # labels: point belongs to cluster
print (labels.shape)
print (labels[0:10])
xb_encoded = index.pq.compute_codes(xb) # xb_encoded: n * 40
print (xb_encoded.shape)

cluster_points = [] # cluster -> points
for i in range(nlist):
    cluster_points.append([])
for i in range(N):
    cluster_points[labels[i]].append(i)

cluster_entry_points = [] # cluster, dim, entry -> points
for i in range(nlist):
    cluster_entry_points.append([])
    for j in range (pq_d):
        cluster_entry_points[i].append ([])
        for k in range (256):
            cluster_entry_points[i][j].append ([])
for i in range (N):
    for j in range (pq_d):
        cluster_entry_points[labels[i]][j][xb_encoded[i][j]].append (i)

q = 100
random_indices = []
while len(random_indices) < q:
    index = np.random.randint(0, 10000, 1)[0]
    if index not in random_indices:
        random_indices.append(index)
xq = [xq[ind] for ind in random_indices]

def IP(x, y):
    return np.dot (x, y)

recall_1_100 = 0
recall_100 = []
for qid in range (q):
    query = xq[qid]
    centroid_id = list (range (0, 1000))
    centroid_id.sort (key = lambda x: IP (query, centroids[x]), reverse=True)
    centroid_id = centroid_id[0 : nprobe]

    counter = {}
    for d in range (pq_d):
        for cluster_id in centroid_id:
            for entry_id in range (256):
                entry_point = centroids[cluster_id][d * M : (d + 1) * M] + codebook[d][entry_id]
                # print (entry_point, IP (query[d * M : (d + 1) * M], entry_point))
                if IP (query[d * M : (d + 1) * M], entry_point) > 0.07:
                    for point_id in cluster_entry_points[cluster_id][d][entry_id]:
                        if point_id in counter:
                            counter[point_id] += 1
                        else:
                            counter[point_id] = 1
    counter_list = []
    for point in counter:
        counter_list.append ((point, counter[point]))
    counter_list.sort (key = lambda x: x[1], reverse=True)
    print (counter_list[0:10])
    counter_list = [x[0] for x in counter_list]
    # print (list (map (IP, [query] * 10, [xb[ind] for ind in counter_list[0:10]])))
    # print (list (map (IP, [query] * 10, [xb[ind] for ind in gt[random_indices[qid]][0:10]])))
    top100 = gt[random_indices[qid]][0:100]
    if top100[0] in counter_list[0:100]:
        recall_1_100 += 1
    cnt100 = 0
    for i in range (min (1000), len (counter_list)):
        if counter_list[i] in top100:
            cnt100 += 1
    recall_100.append (cnt100 / 100)

print (recall_1_100 / q)
print (np.mean (recall_100))