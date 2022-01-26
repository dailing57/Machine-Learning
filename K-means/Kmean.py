import random
import matplotlib.pyplot as plt
import numpy as np

ex = np.loadtxt('ex4x.dat')
TOTAL_NUM = 80
DIMENSION = 2
K_NUM = 2

minn = [ex[:,0].min(),ex[:,1].min()]
maxn = [ex[:,0].max(),ex[:,1].max()]
k_dots = []
colors = ["blue","red","green","yellow"]
for i in range(K_NUM):
    k_dots.append([random.uniform(minn[0],maxn[0]),random.uniform(minn[1],maxn[1])])

cnt = 0
last_sum = -1
cur0 = 0
while last_sum != cur0:
    last_sum = cur0
    cur_sum = np.zeros((K_NUM, DIMENSION))
    tags = np.zeros(K_NUM)
    for i in range(K_NUM):
        plt.scatter(k_dots[i][0],k_dots[i][1],color=colors[i],marker='x')
    for i in range(TOTAL_NUM):
        cur_dist = 1e9
        cur_tag = 0
        for j in range(K_NUM):
            dist = np.linalg.norm(ex[i]-k_dots[j])
            if(dist<cur_dist):
                cur_dist = dist
                cur_tag = j
        tags[cur_tag] += 1
        cur_sum[cur_tag] += ex[i]
        plt.scatter(ex[i][0],ex[i][1],color = colors[cur_tag])
    for i in range(K_NUM):
        next_center = cur_sum[i]/tags[i]
        k_dots[i] = next_center
    cur0 = tags [0]
    plt.show()
    print("第",cnt,"次的中心点：")
    print(k_dots)
    print("各个点数量：")
    print(tags)
