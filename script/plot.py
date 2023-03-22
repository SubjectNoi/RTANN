import numpy as np
import matplotlib.pyplot as plt 

def l2(x, y):
    res = 0.0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** 0.5

def calccnt(pts, query, pq_d, pq_m, thres, max_dist):
    cnt = 0
    for i in range(pq_d):
        bound = max_dist[i]
        x = pts[i * pq_m : (i + 1) * pq_m]
        y = query[i * pq_m : (i + 1) * pq_m]
        dis = l2 (x, y)
        if dis < thres * max_dist[i]:
            cnt += 1
            # cnt += thres * max_dist[i] - dis
    return cnt
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

d = 128
n = 1000000
nlists = 1000
q = 100
radius = 0.3

xb = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_base.fvecs")
xq = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_query.fvecs")

print ("base and query read finished.")

stat = []
for i in range(d):
    tmp = [x[i] for x in xb]
    stat.append([i, np.min(tmp), np.max(tmp), np.mean(tmp), np.std(tmp)])

pq_m = 2
pq_d = int(d // pq_m)
max_dist = []
for i in range(pq_d):
    res = 0.0
    for j in range(pq_m):
        res += (stat[i * pq_m + j][2] ** 2)
    max_dist.append(res ** 0.5)


for idx in range (0, 3):
    query = xq[idx]
    cnt, L2dist = [], []
    for item in xb:
        cnt.append(calccnt(item, query, pq_d, pq_m, radius, max_dist))
        L2dist.append(l2(item, query))

    plt.scatter (L2dist, cnt, s=1)
    plt.savefig ("dist_cnt" + str (idx) + ".png")