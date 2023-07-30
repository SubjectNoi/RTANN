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
res = faiss.StandardGpuResources()

index = faiss.index_factory(D, "IVF1000,PQ40")
co = faiss.GpuClonerOptions()
co.useFloat16 = True
index = faiss.index_cpu_to_gpu(res, 0, index, co)
M = 5
pq_d = D // M
index.train(xb)
index.add(xb)

codebook = faiss.contrib.inspect_tools.get_pq_centroids(index.pq)
print(codebook.shape)
centroids = index.quantizer.reconstruct_n(0, index.nlist) 
get_label_index = faiss.IndexFlat(D)
get_label_index.train(centroids)
get_label_index.add(centroids)
labels = get_label_index.search(x=xb, k=1)[1].reshape(-1)
print (labels.shape)
print (labels[0:10])
xb_encoded = index.pq.compute_codes(xb)
print (xb_encoded.shape)
'''
for d in range(64):
    X, Y = [codebook[d][i][0] + centroids[0][2*d] for i in range(256)], [codebook[d][i][1] + centroids[0][2*d+1] for i in range(256)]
    plt.scatter(X, Y)
    plt.savefig("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/heatmap/Codebook_%d.png" % (d), dpi=600)
    plt.clf()
'''

for d in range(pq_d):
    bias = np.array(centroids[0][M*d:M*(d+1)])
    # biasX, biasY = centroids[0][2*d], centroids[0][2*d+1]
    for c in range(256):
        # codebook[d][c][0] += biasX
        # codebook[d][c][1] += biasY
        codebook[d][c] += bias


hist, _ = np.histogram(labels, bins=index.nlist)
# fig = plt.figure(figsize=(8, 1.5))
plot_q = 100
total_counter = []
def L2(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
def IP(x, y):
    res = 0.0
    for i in range(len(x)):
        res += x[i] * y[i]
    return res
fig = plt.figure(figsize=(4,3))
# for q in [51, 54, 57, 63]:
total_accum = []
for q in range(plot_q):
# for q in [4, 7, 25, 31]: # Deep
# for q in [51, 54, 57, 63]: # Sift
    # mat = np.zeros((64, 256))
    gtq = gt[q]
    query = xq[q]
    different = []
    query_heatmap = []
    for d in range(pq_d):
        gt_code = [xb_encoded[i][d] for i in gtq]
        q_seg = query[M*d:M*(d+1)]
        IN, OUT = [], []
        ALL = []
        for c in range(256):
            if c in gt_code:
                IN.append(IP(q_seg, codebook[d][c]))
                ALL.append([c, gt_code.count(c), IP(q_seg, codebook[d][c])])
            else:
                OUT.append(IP(q_seg, codebook[d][c]))
                ALL.append([c, 0, IP(q_seg, codebook[d][c])])
        # print(IN)
        # print(OUT)
        ALL.sort(key=lambda x : x[2])
        ALL = ALL[::-1]
        query_heatmap.append([x[1] for x in ALL])
        # different.append(np.mean(IN) - np.mean(OUT))
        # plt.hist(IN, bins=10, alpha=0.5, label="Selected")
        # plt.hist(OUT, bins=10, alpha=0.5, label="Not selected")
        # plt.legend()
        # plt.savefig("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/heatmap/Distrib_%d.png" % (d), dpi=600)
        # plt.clf()
    non_zero = 0
    for d in range(pq_d):
        for c in range(256):
            if query_heatmap[d][c] > 3: non_zero += query_heatmap[d][c]
    print(q, non_zero)
    accum = []
    line_95, line_99 = -1, -1
    local_non_zero = 0
    for c in range(256):
        for d in range(pq_d):
            if query_heatmap[d][c] > 3: local_non_zero += query_heatmap[d][c]
        factor = (1.0 * local_non_zero) / (1.0 * non_zero)
        if factor > 0.95 and line_95 == -1:
            line_95 = c
        if factor > 0.99 and line_99 == -1:
            line_99 = c
        accum.append((1.0 * local_non_zero) / (1.0 * non_zero))
    new_query_heatmap = np.zeros((pq_d, 256))
    for d in range(pq_d):
        for c in range(256):
            if query_heatmap[d][c] != 0:
                new_query_heatmap[max(0, d-1)][max(0, c-1)]     += query_heatmap[d][c]
                new_query_heatmap[max(0, d-1)][c]               += query_heatmap[d][c]
                new_query_heatmap[max(0, d-1)][min(255, c+1)]   += query_heatmap[d][c]
                new_query_heatmap[d][max(0, c-1)]               += query_heatmap[d][c]
                new_query_heatmap[d][c]                         += query_heatmap[d][c]
                new_query_heatmap[d][min(255, c+1)]             += query_heatmap[d][c]
                new_query_heatmap[min(pq_d-1, d+1)][max(0, c-1)]    += query_heatmap[d][c]
                new_query_heatmap[min(pq_d-1, d+1)][c]              += query_heatmap[d][c]
                new_query_heatmap[min(pq_d-1, d+1)][min(255, c+1)]  += query_heatmap[d][c]
    total_accum.append(accum)

'''
    sns.heatmap(new_query_heatmap, cmap="Reds", yticklabels=[], xticklabels=[], linewidth=0.01, linecolor='white', cbar=False)
    plt.savefig("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/heatmap/hm_%d.eps" % (q), format="eps", dpi=300)
    plt.clf()
    # plt.plot(accum, linewidth=1, c='red')
'''
mean, quantile25, quantile75 = [], [], []
mmin, mmax = [], []
for c in range(256):
    tmp = [x[c] for x in total_accum]
    mean.append(np.median(tmp))
    q1 = np.quantile(tmp, 0.25)
    q3 = np.quantile(tmp, 0.75)
    quantile25.append(q1)
    quantile75.append(q3)
    iqr = q3-q1
    mmin.append(q1 - 1.5 * iqr)
    mmax.append(q3 + 1.5 * iqr)
c1 = "#90A955"
c2 = "#4F772D"
plt.fill_between(range(256), mmin, mmax, color=c2, alpha=0.5)
plt.plot(mmin, linewidth=0.25, c='black', linestyle='--', label="                               ")
plt.plot(mmax, linewidth=0.25, c='black', linestyle='--')
plt.plot(quantile25, linewidth=0.5, c=c1, linestyle='-.', label="                               ")
plt.plot(quantile75, linewidth=0.5, c=c1, linestyle='-.')
plt.plot(mean, c=c1, linewidth=1, label="                               ")
plt.xticks([0, 64, 128, 192, 256], [])
plt.yticks([0, 0.5, 1.0], ["", "", ""])
plt.xlim(0, 256)
plt.ylim(0.0, 1.0)
    # plt.plot([line_95, line_95], [0.0, 0.95], linewidth=1, linestyle='--', c='grey', alpha=0.5)
    # plt.plot([line_99, line_99], [0.0, 0.99], linewidth=1, linestyle='--', c='grey', alpha=0.5)
plt.plot([0, 255], [0.9, 0.9], linewidth=1.0, linestyle='--', c='black')
plt.legend()
plt.savefig("pdf_all_tti1m_dim3.pdf", format='pdf', dpi=300)
plt.clf()
    # print("Query: %05d, Different: %7.3f" % (q, np.mean(different)))

'''
total_counter = []
plot_q = 100
figure = plt.figure(figsize=(2.8, 2.1))
for q in range(plot_q):
    mat = np.zeros((D // 2, 256))
    counter = [0] * (D // 2)
    gtq = gt[q]
    xbq = [xb[i] for i in gtq]
    xbq = np.array(xbq)
    codes = index.pq.compute_codes(xbq)
    for item in codes:
        for i in range(len(item)):
            mat[i][item[i]] += 1
    # print(mat)                                                                                                                         
    # ax = sns.heatmap(mat, cmap="Reds", yticklabels=[], xticklabels=[], linewidth=0.02, linecolor='grey')
    # plt.ylabel("Dim ID")
    # plt.xlabel("Codebook Entry ID")
    # plt.savefig("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/heatmap/Query%d_Heatmap.png" % (q), dpi=600)    
    # plt.clf()
    for d in range(D // 2):
        cnt = 0
        for c in range(256):
            if mat[d][c] != 0:
                cnt += 1
        counter[d] = cnt
    total_counter.append(counter)
counter_per_dim = []
for c in range(D // 2):
    tmp = []
    for item in total_counter:
        tmp.append((item[c] * 1.0) / 256.0)
    counter_per_dim.append(tmp)
# plt.ylim(0, 1.0)
# plt.boxplot(counter_per_dim)
max_cpd = []
mean_cpd = []
for item in counter_per_dim:
    max_cpd.append(np.max(item))
    mean_cpd.append(np.mean(item))
plt.ylim(0, 0.4)
plt.xlim(0, D // 2)
# plt.scatter(range(64), max_cpd, s=10, , label="max used entry ratio")
plt.plot(max_cpd, c='red', label="                    ")
plt.plot(mean_cpd, c='#FFC0CB', label="                   ")
plt.legend()
plt.xticks([0, 8, 16, 24, 32, 40, 48], ["", "", "", "", "", "", ""])
plt.yticks([0.0, 0.1, 0.2, 0.3], ["", "", "", ""])
plt.savefig("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/heatmap/box.eps", format="eps", dpi=300)


'''    
