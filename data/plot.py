import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
fcodebook = open("codebook_toy.txt").readlines()
fquery = open("query_toy.txt").readlines()
# forigin = open("origin.txt").readlines()
n_codebook_entry = 4
n_dim = 8
n_codebook = n_dim // 2

Coord = []
Pivot = []
OriginX = []
OriginY = []
for i in range(n_codebook):
    X, Y = [], []
    for j in range(n_codebook_entry):
        tmp = [float(x) for x in fcodebook[i * n_codebook_entry + j].split("\n")[0].split()]
        X.append(tmp[0])
        Y.append(tmp[1])
    Coord.append([X, Y])

q = [float(x) for x in fquery[0].split("\n")[0].split()]
for i in range(n_codebook):
    Pivot.append([q[i * 2], q[i * 2 + 1]])


# for i in range(len(forigin)):
#     tmp = [float(x) for x in forigin[i].split("\n")[0].split()]
#     OriginX.append(tmp[0])
#     OriginY.append(tmp[1])

for i in range(n_codebook):
    fig, axes = plt.subplots(figsize=(10, 10))
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.scatter(Coord[i][0], Coord[i][1], s=50, label='Codebook Centroids', marker='x')
    plt.scatter(Pivot[i][0], Pivot[i][1], marker='^', label='Query')
    rect = plt.Rectangle((Pivot[i][0]-0.1, Pivot[i][1]-0.1), 0.2, 0.2, fc='white', ec='red', alpha=0.5)
    plt.gca().add_patch(rect)
    # cir = plt.Circle((Pivot[i][0], Pivot[i][1]), 0.05, fill=False)
    # axes.add_artist(cir)
    # plt.scatter(OriginX, OriginY, s=5, marker='.', label='Original Points', alpha=0.5)
    plt.legend()
    plt.savefig("Fig_%d.png" % (i), dpi=1200)
    plt.clf()
