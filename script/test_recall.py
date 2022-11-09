from math import floor
import numpy as np
from os import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import time
tt = float(sys.argv[1])
def L2(x, y):
    res = 0.0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** 0.5

def CalcCnt(x, y, thres, stat):
    res = 0
    for i in range(0, len(x), 2):
        distl2 = ((x[i] - y[i]) ** 2 + (x[i + 1] - y[i + 1]) ** 2) ** 0.5
        distl0 = max(x[i] - y[i], x[i + 1] - y[i + 1])
        distl1 = abs(x[i] - y[i]) + abs(x[i + 1] - y[i + 1])
        # dist = dist ** 0.5
        dist = distl2
        factor = (stat[i][2] - stat[i][1]) ** 2 + (stat[i + 1][2] - stat[i][1]) ** 2
        factor = factor ** 0.5
        if dist < thres * factor:
            res += 1
        if dist > tt * factor:
            res -= 0.5
    return res

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def Calculation(cluster_num, thres):
    recalls = []
    for i in range (q):
        queryIdx = floor (np.random.random() * len (xq))
        query = xq[queryIdx]

        # Pick the closest centroid and corresponding cluster
        ccs.sort(key=lambda x : L2(query, x[1]))
        lab = [x[0] for x in ccs]

        # cluster_num control how many cluster I picked
        lab = lab[0:cluster_num]

        candidate = []

        # Picked the point with selected cluster's label
        for clusterIdx in lab:
            candidate = candidate + labelToPoint[clusterIdx]
        # for i in range(len(xb)):
        #     if labels[i] in lab:
        #         candidate.append([i, xb[i]])

        candidate_ = []

        for x in candidate:
            # Calculate the occurance of coord-pair with L2<thres
            c = CalcCnt(x[1], query, thres, stat)
            candidate_.append((x[0], c))

        # Sort by the occurance
        candidate_.sort(key=lambda x : x[1])

        # Pick Top 1000
        candidate_ = candidate_[::-1][0:1000]
        # Return the index of top 1000
        result = [x[0] for x in candidate_]
    
        # Now compute the ground truth with the most stupid method
        # Sort all point via L2 dist with query
        xb_.sort(key=lambda x : L2(query, x[1]))
        # Pick true top 100
        Top100 = xb_[0:100]
        # Get index of true top 100
        ref = [int(x[0]) for x in Top100]

        # Calculate Recall: R100@1000
        recall = 0
        for r in result:
            if r in ref:
                recall += 1
        # print(thres, recall)
        recalls.append(recall)
    # print(recalls)
    # print(np.mean(recalls), np.var(recalls)**0.5)
    return recalls

d = 128
n = 131072
nlists = 16
q = 10
cluster_num = 1

filename = "recall"
# filepath = '/home/wtni/RTANN/' 
filepath = '/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/script/' + filename
# np.random.seed(int(time.time()))
np.random.seed(7752)

# xb = np.random.rand(n, d).astype("float32")
xb = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_base.fvecs")
startIdx = floor (np.random.random() * (len(xb) - n))
print ("idx " + str(startIdx) + " to " + str (startIdx + n))
xb = xb[startIdx : startIdx + n]
stat = []

# xb = normalize (xb)

# Normalize the data to (0,1), every dimension uses its own max to normalize
for i in range(len(xb[0])):
    tmp = [x[i] for x in xb]
    stat.append([i, np.min(tmp), np.max(tmp), np.mean(tmp), np.std(tmp)])

# for i in range(len(xb[0])):
#     for j in range(len(xb)):
#         xb[j][i] /= stat[i][2]
# xb = normalize (xb)

kmeans = KMeans(n_clusters=nlists, init='k-means++', n_init=4).fit(xb)

# xq = np.random.rand(q, d).astype("float32")
xq = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_query.fvecs")

# Normalize query 
# for i in range(len(xq[0])):
#     for j in range(len(xq)):
#         xq[j][i] /= stat[i][2]
# xq = normalize (xq)

cluster_centroids = kmeans.cluster_centers_
labels = kmeans.labels_

labelToPoint = []
for idx in range (len (cluster_centroids)):
    labelToPoint.append ([])
for idx in range (len (xb)):
    labelToPoint[labels[idx]].append ((idx, xb[idx]))

xb_ = []
for i in range(len(xb)):
    xb_.append((i, xb[i]))

ccs = []
for i in range(len(cluster_centroids)):
    ccs.append((i, cluster_centroids[i]))

print ("Clustering Finished.")

res = []

for idx in range (1, 11):
    thres = idx / 20
    print ("thres=" + str (thres))
    queryStartIdx = floor (np.random.random() * (len (xq) - q))
    # curQ = xq[queryStartIdx : queryStartIdx + q]
    # print ("cur query idx: " + str (queryStartIdx) + " to " + str (queryStartIdx + q))
    recalls = Calculation(cluster_num, thres)
    res.append(recalls)
print('')
# np.save (filepath + '.npy', res)
for r in res:
    print(r)
fig, ax = plt.subplots()
figPos = []
figLabels = []
for idx in range (1, 11):
    figPos.append (idx * 4)
    figLabels.append ('%.1f' % (idx * 0.05))

VP = ax.boxplot(res, positions=figPos, widths=2, patch_artist=True,
                showmeans=False, showfliers=False, labels=figLabels, 
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})
ax.set_ylim(0, 100)
plt.savefig (filepath + '_tt=%f.png' % (tt), dpi=1200)