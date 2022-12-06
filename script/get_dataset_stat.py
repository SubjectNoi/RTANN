import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

f = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/SIFT1M/search_points").readlines()

xb = []

for item in f:
    xb.append([float(x) for x in item.split()])

for d in range(len(xb[0])):
    tmp = [x[d] for x in xb]
    _min, _max, _mean, _median, _std = np.min(tmp), np.max(tmp), np.mean(tmp), np.median(tmp), np.std(tmp)
    print(_max, end=', ')