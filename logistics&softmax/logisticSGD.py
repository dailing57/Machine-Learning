import random

import numpy as np

ex = np.loadtxt('ex4x.dat')
ey = np.loadtxt('ex4y.dat')
TRAIN_NUM = 70
TOTAL_NUM = 80
DIMENSION = 3
ex = np.insert(ex, 0, values=np.ones(TOTAL_NUM), axis=1)
train_x = ex[:TRAIN_NUM]
train_y = ey[:TRAIN_NUM]
test_x = ex[TRAIN_NUM:TOTAL_NUM]
test_y = ey[TRAIN_NUM:TOTAL_NUM]


def h(theta, x):
    return 1/(1+np.exp(-np.inner(theta, x)))


def loss(theta, x, y):
    res = 0
    sz = np.size(x, 0)
    for i in range(sz):
        res += y[i]*np.log(h(theta, x[i]))+(1-y[i])*np.log(1-h(theta, x[i]))
    return -res/sz


alpha = 5e-5
theta = np.ones(DIMENSION)

for i in range(50000):
    sum = np.zeros(DIMENSION)
    id = random.randint(0,TRAIN_NUM-1)
    sum = sum + (train_y[id]-h(theta, train_x[id]))*train_x[id]
    theta = theta + alpha/40*sum
    # print(loss(theta, train_x, train_y))

print(theta)
print(loss(theta, test_x, test_y))

TP = 0
TN = 0
FP = 0
FN = 0
for i in range(TOTAL_NUM-TRAIN_NUM):
    if(h(theta,test_x[i])<0.5):
        if test_y[i]==0:
            TN+=1
        else:
            FN+=1
    else:
        if test_y[i]==1:
            TP+=1
        else:
            FP+=1

if(TP+FP!=0):
    print("positive precision:",TP/(TP+FP))

if(TP+FN!=0):
    print("positive recall:",TP/(TP+FN))

if TN+FN!=0:
    print("negative precision:",TN/(TN+FN))
if TN+FP!=0:
    print("negative recall:",TN/(TN+FP))

