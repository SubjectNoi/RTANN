import numpy as np
import sys
from cuml import KMeans

d = 200
n = 1000000
m = 2
pq_d = d // m
nlists = int(sys.argv[1])
pq_nlists = int(sys.argv[2])

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
 
def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def IP(x, y):
    res = 0.0
    for i in range(len(x)):
        res += x[i] * y[i]
    return res

xb  = read_fbin("/home/zhliu/workspace/TTI1M/base.1M.fbin")
xq  = read_fbin("/home/zhliu/workspace/TTI1M/query.public.100K.fbin", 0, 10000)

xb_l2, xq_l2 = [], []
cnt = 0
for x in xb:
    _x = [val for val in x]
    _x.append((10000 - np.sum(np.array(_x) ** 2)) ** 0.5)
    xb_l2.append(_x)
    print("%8d / %8d" % (cnt, n), end='\r')
    cnt += 1

cnt = 0
for x in xq:
    _x = [val for val in x]
    _x.append(0.0)
    xq_l2.append(_x)    
    print("%8d / %8d" % (cnt, 10000), end='\r')
    cnt += 1

xb_l2, xq_l2 = np.array(xb_l2), np.array(xq_l2)
print(xb_l2.shape, xq_l2.shape)
kmeans = KMeans(n_clusters=nlists, init='scalable-k-means++', n_init=1).fit(xb_l2)
cluster_centroids = kmeans.cluster_centers_
labels = kmeans.labels_
f1 = open("/home/wtni/RTANN/RTANN/data/TTI1M/parameter_correct/cluster_centroids_%d" % (nlists), "w+")
for cc in cluster_centroids:
    for i in range(200):
        f1.write("%f " % (cc[i]))
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

for keys in cluster_mapping:
    xb_of_this_cluster = [xb[i] for i in cluster_mapping[keys]]
    for d in range(pq_d):
        X = [[x[2*d], x[2*d+1], (100 - x[2*d]**2 - x[2*d+1]**2) ** 0.5] for x in xb_of_this_cluster]