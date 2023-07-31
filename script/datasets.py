# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import time
import numpy as np

path1m = "/home/zhliu/workspace/faiss_sample/"
# path = "/home/zhliu/SIFT1B/"
path = "/home/zhliu/workspace/TTI1M/"

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
 

def ivecs_read(fname, cnt=-1):
    a = np.fromfile(fname, dtype='int32', count=cnt)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, cnt=-1):
    return ivecs_read(fname).view('float32')

def bvecs_read(fname, cnt=-1):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8, count=cnt)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy().astype("float32")

def load_tti1M():
    print("Loading tti1M...", end='', file=sys.stderr)
    xt = []
    xb = read_fbin(path + "base.1M.fbin")
    xq = read_fbin(path + "query.public.100K.fbin", 0, 10000)
    # gt = read_ibin(path + "groundtruth.public.100K.ibin", 0, 10000)
    gt = []
    # f = open("/home/zhliu/workspace/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/RTANN/data/TTI1M/ground_truth", "r").readlines()
    f = open("/home/zhliu/workspace/TTI1M/groundtruth_1M", "r").readlines()
    for item in f:
        tmp = [int(x) for x in item.split()]
        gt.append(tmp)
    print("done", file=sys.stderr)
    return xb, xq, xt, np.array(gt)

def load_sift1B():
    print("Loading sift1B...", end='', file=sys.stderr)
    # xt = bvecs_read(path + "bigann_learn.bvecs", cnt=132000000)
    xb = bvecs_read(path + "bigann_base.bvecs", cnt=13200000000)
    xt = []
    xq = bvecs_read(path + "bigann_query.bvecs")
    gt = ivecs_read(path + "gnd/idx_100M.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read(path1m + "sift/sift_learn.fvecs")
    xb = fvecs_read(path1m + "sift/sift_base.fvecs")
    xq = fvecs_read(path1m + "sift/sift_query.fvecs")
    gt = ivecs_read(path1m + "sift/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt

def load_deep1M():
    # print("Loading deep1M...", end='', file=sys.stderr)
    xt = fvecs_read(path1m + "deep/deep1M_base.fvecs", cnt=100000)
    xb = fvecs_read(path1m + "deep/deep1M_base.fvecs")
    xq = fvecs_read(path1m + "deep/deep1B_queries.fvecs")
    gt = ivecs_read(path1m + "deep/deep1M_groundtruth.ivecs")
    # print("done", file=sys.stderr)

    return xb, xq, xt, gt

def load_deep1B():
    path = "/home/wtni/RTANN/RTANN/data/Deep1B/"
    print("Loading deep1B...", end='', file=sys.stderr)
    xt = fvecs_read(path + "deep10M.fvecs")
    # xb = fvecs_read(path + "base.fvecs")

    array_list = []
    d = 96
    for i in range (4):
        a = np.fromfile(path + "base_0" + str (i), dtype='int32', count=-1)
        print (a.shape)
        array_list.append (a)

    n = 100 * 1000000 # 100M
    xb = np.concatenate (array_list)[:97 * n].reshape (-1, d + 1).copy()

    xq = fvecs_read(path + "deep1B_queries.fvecs")
    gt = ivecs_read(path + "deep1B_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt


def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()
    r100_1000 = 0.0
    
    for q in range(nq):
        gt100 = gt[q][0:100]
        cnt = 0
        for g in I[q][0:1000]:
            if g in gt100:
                cnt += 1
        r100_1000 += (1.0 * cnt) / (100.0)
    
    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10
    recalls[1000] = r100_1000 / (1.0 * nq)
    return (t1 - t0) * 1000.0, recalls
