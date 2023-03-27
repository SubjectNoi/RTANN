import matplotlib.pyplot as plt

# f = open("sift1m_res").readlines()
f = open("rt_evaluation.txt").readlines()

d = 128

res = {}
comp_ratio = 1
r1_100, r100_1000 = [], []
qps_r1_100, qps_r100_1000 = [], []
cnt = 0
# colors = ['#EB5353', '#F9D923', '#36AE7C', '#187498']
# colors = ['#4870F0', '#4BA3FA', '#4FC1E3', '#4BFAF4', '#48F0BB']
colors = ['#8ecae6', '#219ebc', '#023047', '#ffb703', '#fb8500', '#d5bdaf']
markers = ['^', '.', 'H', 'D', 'v', '*']
ms = [5, 10, 6.5, 5, 5, 6]
label_ = ""
for item in f:
    tmp = item.split()
    if len(tmp) == 1:
        # comp_ratio = d // int(tmp[0][2:])
        label_ = tmp[0]
        r1_100, r100_1000, qps_r1_100, qps_r100_1000 = [], [], [], []
        continue
    elif len(tmp) == 0:
        print (label_)
        # plt.plot(r1_100, qps_r1_100, label=label_, linewidth=1, c=colors[cnt], marker=markers[cnt], markersize=ms[cnt])
        plt.plot(r100_1000, qps_r100_1000, label=label_, linewidth=1, c=colors[cnt], marker=markers[cnt], markersize=ms[cnt])
        cnt += 1
        continue
    else:
        tmp = [float(x) for x in tmp]
        r1_100.append(tmp[1])
        r100_1000.append(tmp[3])
        qps_r1_100.append(tmp[0])
        qps_r100_1000.append(tmp[2])
plt.yscale('log')
plt.ylim(1000, 10000000)
plt.legend()
plt.savefig ("RT.png", dpi=300)
# plt.show()


