from sklearn.manifold import TSNE
import tmap as tm
import matplotlib.pyplot as plt
from faerun import Faerun
root = "/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN"
fxb = open("%s/data/SIFT1M/search_points" % (root)).readlines()
flb = open("%s/data/SIFT1M/search_points_labels_2000" % (root)).readlines()
dims = 128
lb = [int(x) for x in flb[0].split('\n')[0].split(',')[0:1000000]]
xb = {}
# lb = []
for i in range(len(fxb)):
    item = fxb[i]
    if lb[i] not in xb:
        xb[lb[i]] = []
    xb[lb[i]].append([float(x) for x in item.split('\n')[0].split()])
    # xb.append([float(x) for x in item.split('\n')[0].split()])

cnt = 0
fig = plt.figure(figsize=(20, 20))
for keys in xb:
    enc = tm.Minhash(dims)
    lf = tm.LSHForest(dims, 128)
    lf.batch_add(enc.batch_from_weight_array(xb[keys]))
    lf.index()
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf)

    
    plt.scatter(x, y, s=1)
    for i in range(len(s)):
        plt.plot([x[s[i]], x[t[i]]], [y[s[i]], y[t[i]]], c='black', linewidth=0.25)
    cnt += 1
    if cnt > 4:
        break
    
    
plt.savefig("tsne.png", dpi=1200)

    

