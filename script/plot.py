import numpy as np
import matplotlib.pyplot as plt 

from datasets import load_tti1M, evaluate 

def IP(x, y):
    res = 0.0
    for i in range(len(x)):
        res += x[i] * y[i]
    return res

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
        dis = l2 (x, y)
        if dis < thres * max_dist[i]:
            cnt += 1
            # cnt += thres * max_dist[i] - dis
        # dis = IP (x, y)
        # if dis >= thres * 2 * 0.3 * sum (y):
        #     cnt += 2
        # elif dis >= thres * 0.3 * sum (y):
        #     cnt += 1
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

d = 128
n = 1000000
nlists = 1000
q = 100
radius = 0.5

pq_m = 1
pq_d = int(d // pq_m) * 4

# xb, xq, xt, gts = load_tti1M()
# print (d)

xb = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_base.fvecs")
xq = fvecs_read("/home/zhliu/workspace/faiss_sample/sift/sift_query.fvecs")

nq, d = xq.shape

print ("base and query read finished.")

# random projections
xbb = np.zeros ((n, pq_d * pq_m))
for i in range (pq_d):
    proj = np.random.randn (d, pq_m)
    proj = proj / np.linalg.norm (proj, axis=0)
    xbb[:, i * pq_m : (i + 1) * pq_m] = xb.dot (proj)
xb = xbb

xqq = np.zeros ((nq, pq_d * pq_m))
for i in range (pq_d):
    proj = np.random.randn (d, pq_m)
    proj = proj / np.linalg.norm (proj, axis=0)
    xqq[:, i * pq_m : (i + 1) * pq_m] = xq.dot (proj)
xq = xqq

print ("random projection finished.")

stat = []
for i in range(pq_d * pq_m):
    tmp = [x[i] for x in xb]
    stat.append([i, np.min(tmp), np.max(tmp), np.mean(tmp), np.std(tmp)])
print (stat)

max_dist = []
for i in range(pq_d):
    res = 0.0
    for j in range(pq_m):
        res += (stat[i * pq_m + j][2] ** 2)
    max_dist.append(res ** 0.5)

for idx in range (0, 1):
    query = xq[idx]
    cnt, disList = [], []
    for item in xb:
        cnt.append(calccnt(item, query, pq_d, pq_m, radius, max_dist))
        disList.append (l2 (item, query))
        # disList.append(IP(item, query))

    plt.scatter (disList, cnt, s=1)
    plt.savefig ("IP_l2_randomProjection1D" + ".png")