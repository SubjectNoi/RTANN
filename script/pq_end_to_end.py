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
nlists = int(sys.argv[2])
q = 100
bias = 0
thres = float(sys.argv[1])
cluster_num = int(sys.argv[3])

random_indices = []
while len(random_indices) < q:
    index = np.random.randint(0, 10000, 1)[0]
    if index not in random_indices:
        random_indices.append(index)
# random_indices = [0]

def l2(x, y):
    res = 0.0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** 0.5

def calccnt(pts, query, pq_d, pq_m, thres, max_dist):
    cnt = 0
    for i in range(pq_d):
        # bound = max_dist[i]
        x = pts[i * pq_m : (i + 1) * pq_m]
        y = query[i * pq_m : (i + 1) * pq_m]
        if l2 (x, y) < 1.01:
        # if l2(x, y) <= thres * max_dist[i]:
            cnt += 1
        # dis = IP (x, y) / np.linalg.norm (y)
        # if dis >= thres * 0.42:
        #     cnt += 1
            # cnt += IP (x, y)
        # elif dis >= thres / 2 * 0.3 * sum (y):
        #     cnt += 0
        # else:
        #     cnt += -1
    return cnt
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

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

# xbb = np.zeros ((n, 2 * d))
# for i in range (len (xb)):
#     for j in range (d):
#         xbb[i][j * 2] = (1 - xb[i][j] ** 2) ** 0.5
#         xbb[i][j * 2 + 1] = xb[i][j]
# xb = xbb

# xqq = np.zeros ((nq, 2 * d))
# for i in range (len (xq)):
#     for j in range (d):
#         xqq[i][j * 2] = 0
#         xqq[i][j * 2 + 1] = xq[i][j]
# xq = xqq

nq, d = xq.shape
print (d)
pq_m = 5
pq_d = int(d // pq_m)

# phi: max l2 norm of xb
# print (xb[0].shape, xb[0])
# phi = max (np.linalg.norm(xb, axis=1))

# xbb = []
# for i in range (len (xb)):
#     xbb.append (np.append (xb[i], [(phi ** 2 - np.linalg.norm(xb[i]) ** 2) ** 0.5, 0]))
# xb = np.array (xbb)

# xqq = []
# for i in range (len (xq)):
#     xqq.append (np.append (xq[i], [0, 0]))
# xq = np.array (xqq)
# d += 2

# print (xb[0].shape, xb[0])

print("Reading xb/xq/gt Finished")

# random projections
# xbb = np.zeros ((n, d))
# for i in range (pq_d):
#     proj = np.random.randn (d, pq_m)
#     proj = proj / np.linalg.norm (proj, axis=0)
#     xbb[:, i * pq_m : (i + 1) * pq_m] = xb.dot (proj)
# xb = xbb

# xqq = np.zeros ((nq, d))
# for i in range (pq_d):
#     proj = np.random.randn (d, pq_m)
#     proj = proj / np.linalg.norm (proj, axis=0)
#     xqq[:, i * pq_m : (i + 1) * pq_m] = xq.dot (proj)
# xq = xqq

# print ("random projection finished.")

xb = xb[0:n]
# xq = xq[bias:bias+q]
xq = [xq[ind] for ind in random_indices]
# stat = []
# for i in range(d):
#     tmp = [x[i] for x in xb]
#     stat.append([i, np.min(tmp), np.max(tmp), np.mean(tmp), np.std(tmp)])

# kmeans = KMeans(n_clusters=nlists, init='k-means++', n_init=64).fit(xb)
print ("Start Clustering")
kmeans = KMeans(n_clusters=nlists, init='scalable-k-means++', n_init=32, max_iter=600).fit(xb)
cluster_centroids = kmeans.cluster_centers_
labels = kmeans.labels_
f1 = open("/home/wtni/RTANN/RTANN/data/TTI1M/parameter_PQ40/cluster_centroids=%d" % (nlists), "w+")
for cc in cluster_centroids:
    for x in cc:
        f1.write("%f " % (x))
    f1.write("\n")

f1.write("-----\n")
for l in labels:
    f1.write("%d " % (l))
f1.close()

# cluster_centroids, labels = [], []
# f1 = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1M/parameter_0/cluster_centroids_%d" % (nlists)).readlines()
# f1 = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/parameter_0/cluster_centroids_%d" % (nlists)).readlines()
# f1 = open("/home/wtni/RTANN/RTANN/data/TTI1M/parameter_addDimOnEveryDim/cluster_centroids=%d" % (nlists)).readlines()

# fcentroids, flabels = f1[0:nlists], f1[-1]
# flabels = flabels.split()[0:n]
# for item in fcentroids:
#     tmp = [float(x) for x in item.split()[0:d]]
#     cluster_centroids.append(tmp)
# for item in flabels:
#     labels.append(int(item))
# print("First Cluster Finished")


cluster_centroids_with_id = []
for i in range(len(cluster_centroids)):
    cluster_centroids_with_id.append([i, cluster_centroids[i]])

pq_nlists = int(sys.argv[4])

max_dist = []
# for i in range(pq_d):
#     res = 0.0
#     for j in range(pq_m):
#         res += (stat[i * pq_m + j][2] ** 2)
#     max_dist.append(res ** 0.5)

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
        X = [x[pq * pq_m : (pq + 1) * pq_m] for x in [y[1] for y in current_cluster]]
        X = np.array(X)
        subkmeans = KMeans(n_clusters=pq_nlists, init='scalable-k-means++', n_init=32, max_iter=600).fit(X)
        sub_centroids = subkmeans.cluster_centers_
        sub_labels = subkmeans.labels_
        fcluster = open("/home/wtni/RTANN/RTANN/data/TTI1M/parameter_PQ40/codebook_%d/codebook_cluster=%d_dim=%d" % (nlists, keys, pq), "w+")
        for item in sub_centroids:
            for x in item:
                fcluster.write("%f " % (x))
            fcluster.write("\n")
        fcluster.write("-----\n")
        for item in sub_labels:
            fcluster.write("%d " % (item))
        fcluster.close()

        # fcluster = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1M/parameter_0/codebook_%d/codebook_cluster=%d_dim=%d" % (nlists, keys, pq)).readlines()
        # fcluster = open("/home/wtni/RTANN/RTANN/data/TTI1M/parameter_addDimOnEveryDim/codebook_%d/codebook_cluster=%d_dim=%d" % (nlists, keys, pq)).readlines()
        # fcluster = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/parameter_0/codebook_%d/codebook_cluster=%d_dim=%d" % (nlists, keys, pq)).readlines()
        # sub_centroids, sub_labels = [], []
        # fsubcentroids, fsublabels = fcluster[0:pq_nlists], fcluster[-1]
        # fsublabels = fsublabels.split()[0:len(current_cluster)]
        # for item in fsubcentroids:
        #     tmp = [float(x) for x in item.split()[0:pq_m]]
        #     sub_centroids.append(tmp)
        # for item in fsublabels:
        #     sub_labels.append(int(item))

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

codebook_range = np.zeros((nlists, pq_d))
for i in range(nlists):
    current_codebook = codebook[i]
    for j in range(pq_d):
        current_codebook_at_dim = current_codebook[j]
        x = [e[0] for e in current_codebook_at_dim]
        y = [e[1] for e in current_codebook_at_dim]
        codebook_range[i][j] = ((np.max(x) - np.min(x)) ** 2 + (np.max(y) - np.min(y)) ** 2) ** 0.5

recall_0, recall_1, recall_2 = [], [], []  
cluster_dict = {}
non_zero_count = np.zeros((q, pq_d))
r1cnt, r1pqcnt = 0, 0
r1cnt10, r1pqcnt10 = 0, 0
r1cnt100, r1pqcnt100 = 0, 0
for qid in range(len(xq)):
    query = xq[qid]
    query_sum = sum (query)
    query_seg = []
    for pq in range(pq_d):
        query_seg.append(query[pq * pq_m : (pq + 1) * pq_m])
    cluster_centroids_with_id = []
    for i in range(len(cluster_centroids)):
        cluster_centroids_with_id.append([i, cluster_centroids[i]])
    cluster_centroids_with_id.sort(key=lambda x : l2(query, x[1]))
    cluster_centroids_with_id = cluster_centroids_with_id[0:cluster_num]
    sel_key = [x[0] for x in cluster_centroids_with_id]

    # print(sel_key)
    # Calculate Ground Truth
    gt = [int(x) for x in gts[random_indices[qid]]]
    nearest = gt[0]
    # Calculate Full Recall
    full_candidate = []
    for i in xb_with_2_cluster_info:
        if i[2] in sel_key:
            full_candidate.append(i)

    gt_in_cluster = []
    full_candidate_id = [x[0] for x in full_candidate]
    recall_full = 0
    for fc in full_candidate_id:
        if fc in gt:
            gt_in_cluster.append(fc)
            recall_full += 1
    recall_0.append(recall_full)

    candidate_with_cnt = []
    for fc in full_candidate:
        counter = calccnt(fc[1], query, pq_d, pq_m, thres, max_dist)
        candidate_with_cnt.append([fc, counter])
    candidate_with_cnt.sort(key=lambda x : x[1])
    candidate_with_cnt = candidate_with_cnt[::-1]
    candidate_id = [x[0][0] for x in candidate_with_cnt[0:1000]]
    recall = 0
    for c in candidate_id:
        if c in gt:
            recall += 1
    recall_1.append(recall)
    if nearest in candidate_id[0:100]:
        r1cnt100 += 1
    if nearest in candidate_id[0:10]:
        r1cnt10 += 1
    if nearest == candidate_id[0]:
        r1cnt += 1
    
    occurance = {}
    print(sel_key)
    for skey in sel_key:
        current_codebook = codebook[skey]
        for sid in range(len(query_seg)): # assert len(query_seg) == pq_d
            seg_candidate = []
            seg = query_seg[sid]
            codebook_line_with_id = []
            for i in range(len(current_codebook[sid])):
                codebook_line_with_id.append([i, current_codebook[sid][i]])
            # codebook_line_with_id.sort(key=lambda x : IP(x[1], seg), reverse=True)
            selected_codebook_points_num = 0

            for i in range(len(codebook_line_with_id)):
                codebook_point = codebook_line_with_id[i][1]
                weight = 0
                tmp_dist = l2 (seg, codebook_point)
                if tmp_dist < 1.01:
                # if tmp_dist <= thres * max_dist[sid]:
                    weight = 1
                # tmp_dist = IP(seg, codebook_line_with_id[i][1]) / np.linalg.norm (seg)
                # if tmp_dist >= thres * 0.42:
                #     weight = 1
                # if tmp_dist >= thres * 0.3 * sum (seg):
                #     weight = 1
                # elif tmp_dist >= thres / 2 * 0.3 * sum (seg):
                #     weight = 0
                # else:
                #     weight = -1
                sceid = codebook_line_with_id[i][0]
                for point in TREE[skey][sid][sceid]:
                    if point[0] not in occurance:
                        occurance[point[0]] = 0
                    occurance[point[0]] += weight

            # for i in range(len(codebook_line_with_id)):
            #     # tmp_dist = l2(seg, codebook_line_with_id[i][1])
            #     # if tmp_dist < thres:
            #     #     selected_codebook_points_num += 1
            #     tmp_ip = IP (seg, codebook_line_with_id[i][1])
            #     # print (tmp_ip)
            #     if tmp_ip >= thres / 2 * 0.3 * sum (seg):
            #         selected_codebook_points_num += 1
            #     if tmp_ip >= thres * 0.3 * sum (seg):
            #         selected_codebook_points_num += 1
            # if selected_codebook_points_num == 0:
            #     selected_codebook_points_num = 1
            
            # fig, axes = plt.subplots(figsize=(8,8))
            # plt.scatter([x[1][0] for x in codebook_line_with_id], [x[1][1] for x in codebook_line_with_id], s=5)
            # circle = plt.Circle((seg[0], seg[1]), thres * codebook_range[skey][sid], alpha=0.3)
            # plt.scatter([seg[0]], [seg[1]], s=50, marker='x')
            # axes.add_artist(circle)
            # plt.xlim(0, 100)
            # plt.ylim(0, 100)
            # plt.savefig("query_1_hit_demo/dim=%d.png" % (sid), dpi=600)
            # plt.clf()
            
            # print("%08x" % (mask), end=' ')
            # sel_codebook_entry_id = [x[0] for x in codebook_line_with_id[0:selected_codebook_points_num]]
            
            # mask = 0
            # for sceid in sel_codebook_entry_id:
            #     mask |= (1 << sceid)
            # for sceid in sel_codebook_entry_id:
            #     mask |= (1 << sceid)
            #     for i in TREE[skey][sid][sceid]:
            #         # if i[0] == 1884: print("%2d, %2d,  %08x, %08x, %s" % (sid, sceid, mask, (mask & ((1 << sceid))), "Hit" if ((mask & ((1 << sceid))) != 0) else "   "))
            #         seg_candidate.append(i[0])
            # # print("%08x" % (mask), end=' ')
            # # if sid % 16 == 15: print()
            # for sc in seg_candidate:
            #     if sc not in occurance:
            #         occurance[sc] = 0
            #     occurance[sc] += 1
        # print()
    occurance_list = []
    for keys in occurance:
        occurance_list.append([keys, occurance[keys]])
    # occurance_list.sort(key=lambda x : x[0])
    # for ol in occurance_list:
    #     print(ol[0], ol[1])
    occurance_list.sort(key=lambda x : x[1])
    occurance_list = occurance_list[::-1]
    print (occurance_list[0:10])
    res_pq = [x[0] for x in occurance_list[0:1000]]
    recall_pq = 0
    for r in res_pq:
        if r in gt:
            recall_pq += 1
    recall_2.append(recall_pq)
    if len (res_pq) >= 100 and nearest in res_pq[0:100]:
        r1pqcnt100 += 1
    if len (res_pq) >= 10 and nearest in res_pq[0:10]:
        r1pqcnt10 += 1
    if len (res_pq) >= 1 and nearest == res_pq[0]:
        r1pqcnt += 1
print("\nCase: nprobs=%d, radius=%.6f, query_size=%d" % (cluster_num, thres, q))
print("R100@1000[Exhaust, Radius, PQ]: ", np.mean(recall_0), np.mean(recall_1), np.mean(recall_2))
print("R1@1     [         Radius, PQ]: ", (100.0 * r1cnt) / (1.0 * q), (100.0 * r1pqcnt) / (1.0 * q)) 
print("R1@10    [         Radius, PQ]: ", (100.0 * r1cnt10) / (1.0 * q), (100.0 * r1pqcnt10) / (1.0 * q)) 
print("R1@100   [         Radius, PQ]: ", (100.0 * r1cnt100) / (1.0 * q), (100.0 * r1pqcnt100) / (1.0 * q))    
print("\n")
    # print(recall_full, recall, recall_pq)

    # Calculate Heatmap
    # sel_codebook = codebook[sel_key[0]]
    # codebook_heatmap = np.zeros((pq_nlists, pq_d))
    # for _d in range(pq_d):
    #     sel_codebook_at_d = sel_codebook[_d]
    #     for gic in gt_in_cluster:
    #         tmp = xb[gic][_d * 2 : _d * 2 + 2]
    #         entry_id, tmin = -1, 1e10
    #         for j in range(pq_nlists):
    #             codebook_dist = ((sel_codebook_at_d[j][0] - tmp[0]) ** 2 + (sel_codebook_at_d[j][1] - tmp[1]) ** 2) ** 0.5
    #             if codebook_dist < tmin:
    #                 entry_id = j
    #                 tmin = codebook_dist
    #         codebook_heatmap[entry_id][_d] += 1
    # for dd in range(pq_d):
    #     query_sub_vec = query[dd * 2 : dd * 2 + 2]
    #     ccnntt = 0
    #     for mm in range(pq_nlists):
    #         if codebook_heatmap[mm][dd] > 0:
    #             ccnntt += 1
    #         dist_entry_to_query = ((sel_codebook[dd][mm][0] - query_sub_vec[0]) ** 2 + (sel_codebook[dd][mm][1] - query_sub_vec[1]) ** 2) ** 0.5
    #     non_zero_count[qid][dd] = ccnntt / (1.0 * pq_nlists)

# Plot codebook usage heatmap
# --------------------------------------------
# means = []
# stds = []
# for d in range(pq_d):
#     tmp = [x[d] for x in non_zero_count]
#     means.append(np.mean(tmp))
#     stds.append(np.std(tmp))

# plt.boxplot(non_zero_count)
# plt.savefig("heatmapcnt_gt1.png", dpi=600)
# print(np.mean(means), np.mean(stds))
# --------------------------------------------

# print(np.mean(recall_0), np.mean(recall_1), np.mean(recall_2))
