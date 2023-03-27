import matplotlib.pyplot as plt

f = open("sift1m_res").readlines()

d = 128

res = {}
comp_ratio = 1
r1_100, r100_1000 = [], []
qps_r1_100, qps_r100_1000 = [], []
cnt = 0
colors = ['#EB5353', '#F9D923', '#36AE7C', '#187498']
markers = ['^', '.', 'H', 'D']
ms = [5, 10, 6.5, 5]
label_ = ""
for item in f:
    tmp = item.split()
    if len(tmp) == 1:
        # comp_ratio = d // int(tmp[0][2:])
        label_ = tmp[0]
        r1_100, r100_1000, qps_r1_100, qps_r100_1000 = [], [], [], []
        continue
    elif len(tmp) == 0:
        plt.plot(r1_100, qps_r1_100, label=label_, linewidth=1, c=colors[cnt], marker=markers[cnt], markersize=ms[cnt])
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
plt.show()


