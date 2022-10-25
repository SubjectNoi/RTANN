import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
d = 128
n = 1000000
nlists = 1000
q = 50
thres = float(sys.argv[1])
cluster_num = 2
def L2(x, y):
    res = 0.0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** 0.5

def CalcCnt(x, y, thres, factors):
    res = 0
    for i in range(0, len(x), 2):
        factor = (factors[i][2] ** 2 + factors[i+1][2]**2) ** 0.5
        # factor = 1
        dist = (x[i] - y[i]) ** 2 + (x[i + 1] - y[i + 1]) ** 2
        dist = dist ** 0.5
        if dist < thres * factor:
            res += 1
    return res

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

np.random.seed(int(time.time()))
# xb = np.random.rand(n, d).astype("float32")
xb = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_base.fvecs")
xb = xb[0:n]
stat = []

# Normalize the data to (0,1), every dimension uses its own max to normalize
for i in range(len(xb[0])):
    tmp = [x[i] for x in xb]
    stat.append([i, np.min(tmp), np.max(tmp), np.mean(tmp), np.std(tmp)])

# for i in range(len(xb[0])):
#     for j in range(len(xb)):
#         xb[j][i] /= stat[i][2]

kmeans = KMeans(n_clusters=nlists, init='k-means++', n_init=32).fit(xb)
# xq = np.random.rand(q, d).astype("float32")
xq = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_query.fvecs")
xq = xq[9076:9076+q]
# Normalize query 
# for i in range(len(xq[0])):
#     for j in range(len(xq)):
#         xq[j][i] /= stat[i][2]
cluster_centroids = kmeans.cluster_centers_
labels = kmeans.labels_
recalls = []
print("Clustering Finished.")

for q in range(len(xq)):
    query = xq[q]

    # Pick the closest centroid and corresponding cluster
    ccs = []
    for i in range(len(cluster_centroids)):
        ccs.append([i, cluster_centroids[i]])
    ccs.sort(key=lambda x : L2(query, x[1]))
    lab = [x[0] for x in ccs]

    # cluster_num control how many cluster I picked
    lab = lab[0:cluster_num]

    candidate = []

    # Picked the point with selected cluster's label
    for i in range(len(xb)):
        if labels[i] in lab:
            candidate.append([i, xb[i]])

    candidate_ = []

    for x in candidate:
        # Calculate the occurance of coord-pair with L2<thres
        c = CalcCnt(x[1], query, thres, stat)
        candidate_.append([x[0], c, x[1]])

    # Sort by the occurance
    candidate_.sort(key=lambda x : x[1])
    # candidate_.sort(key=lambda x : L2(x[2], query))
    # XX = [x[1] for x in candidate_]
    # plt.plot(XX)
    # plt.savefig("Fig_%f_Q=%d.png" % (thres, q), dpi=600)
    # plt.clf()

    # Pick Top 1000
    candidate_ = candidate_[::-1][0:1000]
    # Return the index of top 1000
    result = [x[0] for x in candidate_]
 
    # Now compute the ground truth with the most stupid method
    xb_ = []
    for i in range(len(xb)):
        xb_.append([i, xb[i]])
    # Sort all point via L2 dist with query
    xb_.sort(key=lambda x : L2(query, x[1]))
    # Pick true top 100
    xb_ = xb_[0:100]
    # Get index of true top 100
    ref = [int(x[0]) for x in xb_]

    # Calculate Recall: R100@1000
    recall = 0
    for r in result:
        if r in ref:
            recall += 1
    print(thres, recall)
    recalls.append(recall)
print(np.mean(recalls), np.var(recalls)**0.5)
