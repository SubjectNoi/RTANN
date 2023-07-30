# import torch
import numpy as np
import sys
from cuml import KMeans
# from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.decomposition import PCA                                             
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
# from scipy.spatial.distance import cdist
# from scipy.optimize import linear_sum_assignment
import time
# import xgboost as xgb         
# from kmeans_gpu import KMeans
# import torch    
# from kmeans_pytorch import kmeans

d = 200
n = 1000000
nq = 10000
nlists = 1000
q = 100
bias = 0
cluster_num = 32

def l2(x, y):
    res = 0.0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** 0.5

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


random_indices = []
while len(random_indices) < q:
    index = np.random.randint(0, 10000, 1)[0]
    if index not in random_indices:
        random_indices.append(index)

# xb = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_base.fvecs")
# xq = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_query.fvecs")
# gts = ivecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_groundtruth.ivecs")

# xb = fvecs_read("/home/zhliu/workspace/faiss_sample/deep/deep1M_base.fvecs")
# xq = fvecs_read("/home/zhliu/workspace/faiss_sample/deep/deep1B_queries.fvecs")
# gts = ivecs_read("/home/zhliu/workspace/faiss_sample/deep/deep1M_groundtruth.ivecs")
# for i in range(len(xb)):
#     for j in range(d):
#         xb[i][j] *= 100.0

# for i in range(len(xq)):
#     for j in range(d):
#         xq[i][j] *= 100.0

from datasets import load_tti1M, evaluate 
def IP(x, y):
    return np.dot (x, y)
xb, xq, xt, gts = load_tti1M()

nq, d = xq.shape
print (d)
pq_m = 5
pq_d = int(d // pq_m)

print("Reading xb/xq/gt Finished")
xb = xb[0:n]
xq = [xq[ind] for ind in random_indices]

print ("Start Clustering")
cluster_centroids, labels = [], []
# f1 = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/parameter_0/cluster_centroids_%d" % (nlists)).readlines()
f1 = open("/home/wtni/RTANN/RTANN/data/TTI1M/parameter_PQ40/cluster_centroids=%d" % (nlists)).readlines()

fcentroids, flabels = f1[0:nlists], f1[-1]
flabels = flabels.split()[0:n]
for item in fcentroids:
    tmp = [float(x) for x in item.split()[0:d]]
    cluster_centroids.append(tmp)
for item in flabels:
    labels.append(int(item))
print("First Cluster Finished")

cluster_centroids_with_id = []
for i in range(len(cluster_centroids)):
    cluster_centroids_with_id.append([i, cluster_centroids[i]])

pq_nlists = 256

cluster_mapping = {}
for i in range(len(labels)):
    l = labels[i]
    if l not in cluster_mapping:
        cluster_mapping[l] = []
    cluster_mapping[l].append([i, xb[i]])
xb_with_2_cluster_info = []
codebook = {}

for keys in cluster_mapping:
    current_cluster = cluster_mapping[keys]
    pq_centroids, pq_labels = [], []
    for pq in range(pq_d):
        # fcluster = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/parameter_0/codebook_%d/codebook_cluster=%d_dim=%d" % (nlists, keys, pq)).readlines()
        fcluster = open("/home/wtni/RTANN/RTANN/data/TTI1M/parameter_PQ40/codebook_%d/codebook_cluster=%d_dim=%d" % (nlists, keys, pq)).readlines()
        sub_centroids, sub_labels = [], []
        fsubcentroids, fsublabels = fcluster[0:pq_nlists], fcluster[-1]
        fsublabels = fsublabels.split()[0:len(current_cluster)]
        for item in fsubcentroids:
            tmp = [float(x) for x in item.split()[0:pq_m]]
            sub_centroids.append(tmp)
        for item in fsublabels:
            sub_labels.append(int(item))

        pq_centroids.append(sub_centroids)
        pq_labels.append(sub_labels)
    for icc in range(len(current_cluster)):
        cc = current_cluster[icc]
        pid, pt = cc[0], cc[1]
        first_lb = labels[pid]
        second_lb = [x[icc] for x in pq_labels]
        xb_with_2_cluster_info.append([pid, pt, first_lb, second_lb])
    codebook[keys] = pq_centroids
print("Second Cluster Finish")

TREE = {}
ttt = []
for x in xb_with_2_cluster_info:
    if x[2] not in TREE:
        TREE[x[2]] = {}
    for xx in range(len(x[3])):
        if xx not in TREE[x[2]]:
            TREE[x[2]][xx] = {}
        if x[3][xx] not in TREE[x[2]][xx]:
            TREE[x[2]][xx][x[3][xx]] = []
        TREE[x[2]][xx][x[3][xx]].append([x[0], x[1]])
        # 1st cluster, dim, 2nd cluster
for i in range(nlists):
    if i not in TREE:
        print("Error!")
    for j in range(pq_d):
        for k in range(pq_nlists):
            if k not in TREE[i][j]:
                TREE[i][j][k] = []

query = xq[0]
cluster_centroids_with_id = []
for i in range(len(cluster_centroids)):
    cluster_centroids_with_id.append([i, cluster_centroids[i]])
cluster_centroids_with_id.sort(key=lambda x : IP(query, x[1]), reverse=True)
cluster_centroids_with_id = cluster_centroids_with_id[0:cluster_num]
sel_key = [x[0] for x in cluster_centroids_with_id]

# print(sel_key)
# Calculate Ground Truth
gt = [int(x) for x in gts[random_indices[0]]]
top100 = gt[0 : 100]

pq_entries = []
for i in range (pq_d):
    pq_entries.append ([])

top1_entryRank_list = []
for d in range (pq_d):
    for cluster in sel_key:
        for i in range (len (codebook[cluster][d])):
            pq_entries[d].append ([cluster, i, codebook[cluster][d][i]])
        # pq_entries[d] += codebook[cluster][d]
    pq_entries[d].sort (key=lambda x : IP(query[d * pq_m : (d + 1) * pq_m], x[2]), reverse=True)

    IP_list = []
    entry_list = []
    rank_list = []
    entryid_list = []
    cnt_list = []
    cnt = 0
    top1_entryRank = -1
    for i in range (len (pq_entries[d])):
        entry = pq_entries[d][i]
        cluster = entry[0]
        entry_id = entry[1]
        IP_list.append (IP (query[d * pq_m : (d + 1) * pq_m], entry[2]))
        for pair in TREE[cluster][d][entry_id]:
            pointID = pair[0]
            if pointID in top100:
                if pointID == top100[0]:
                    top1_entryRank = i
                entry_list.append (i)
                rank_list.append (top100.index (pointID))
                cnt += 1
        entryid_list.append (i)
        cnt_list.append (cnt)
    if top1_entryRank == -1:
        top1_entryRank = 256
    top1_entryRank_list.append (top1_entryRank)
    print ("Top1 Entry Rank: ", top1_entryRank)

    plt.scatter (entry_list, rank_list, s=1)
    plt.savefig ("entry_pq5/entry_" + str (d) + ".png")
    plt.clf()

    plt.scatter (entryid_list, cnt_list, s=1)
    plt.savefig ("entry_pq5/cnt_" + str (d) + ".png")
    plt.clf()

top1_entryRank_list.sort()
print (top1_entryRank_list[int (0.5 * len (top1_entryRank_list))])
print (top1_entryRank_list[int (0.7 * len (top1_entryRank_list))])
print (top1_entryRank_list[int (0.9 * len (top1_entryRank_list))])
print (top1_entryRank_list[int (0.95 * len (top1_entryRank_list))])