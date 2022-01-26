import numpy as np

ex = np.loadtxt('ex4x.dat')
ey = np.loadtxt('ex4y.dat')
TOTAL_NUM = 80

tmp = np.insert(ex, 0, values=ey, axis=1)
np.random.shuffle(tmp)
ey = tmp[:, 0]
ex = tmp[:, 1:]

NEURON_NUM = [2, 3, 2]
CATEGORIES = 2
W = []
b = []
L = 3

def normalise(x):
    return (x - x.min())/(x.max() - x.min())


# initialise each w
def init():
    ex[:,0] = normalise(ex[:,0])
    ex[:,1] = normalise(ex[:,1])
    for i in range(L - 1):
        w = np.ones((NEURON_NUM[i],NEURON_NUM[i+1]))
        b.append(np.ones(NEURON_NUM[i + 1]))
        W.append(w)


def h1(ww, bb, x):
    return np.inner(ww, x) + bb


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, k):
    sum = 0
    for i in range(CATEGORIES):
        sum += np.exp(x[i])
    return np.exp(x[k]) / sum


alpha = 0.001
def bp(train_x, train_y,isTest):
    # forward propagation
    # layer 1
    a1 = np.ones(NEURON_NUM[1])
    for i in range(NEURON_NUM[1]):
        a1[i] = h1(W[0][:,i], b[0][i], train_x)
    z1 = np.array(list(map(sigmoid, a1)))
    # layer 2
    a2 = np.ones(NEURON_NUM[2])
    for i in range(NEURON_NUM[2]):
        a2[i] = h1(W[1][:,i], b[1][i], z1)
    y0 = softmax(a2, 0)
    y1 = softmax(a2, 1)
    if(isTest==1):
        if (y0 < y1):
            return 1
        return 0
    # back propagation
    # get the current theta2
    y0_real = float(train_y == 0)
    y1_real = float(train_y == 1)
    theta2 = np.array([y0 - y0_real,y1 - y1_real])
    # update the W[1] and b[1] using theta2
    b[1] -= alpha*theta2
    W1_ori = W[1]
    diag = np.diag(z1*(np.ones(NEURON_NUM[1])-z1))
    z1 = np.mat([z1])
    theta2 = np.mat([theta2])
    W[1] -= alpha*np.dot(z1.T, theta2)
    # get the current theta1, using the derivative of g1, original W1, and theta2
    theta1 = np.dot(np.dot(diag,W1_ori),theta2.T)
    # update the W[0] using theta1.T and x
    W[0] -= alpha*np.dot(np.mat(train_x).T,theta1.T)
    b[0] -= alpha*theta1.T.A[0]
    # print(W[0])
    # print(W[1])
    if(y0<y1):
        return 1
    return 0



# 5 fold
GROUP_NUM = 5
K = int(TOTAL_NUM/GROUP_NUM)
def solve():
    pr = 0
    rc = 0
    cnt0 = 0
    cnt1 = 0
    for i in range(K):
        init()
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(TOTAL_NUM):
            baseId = int(j/K)
            if(baseId!=i):
                bp(ex[j],ey[j],0)
            else:
                ans = bp(ex[j],ey[j],1)
                if(ey[j] == 0):
                    if ans == 0:
                        TN += 1
                    else:
                        FN += 1
                else:
                    if ans == 1:
                        TP += 1
                    else:
                        FP += 1
        if(TP+FP!=0):
            cnt1+=1
            pr += TP / (TP + FP)
        if(TP+FN!=0):
            cnt0+=1
            rc += TP / (TP + FN)
    pr /= cnt1
    rc /= cnt0
    print("正确率：",pr)
    print("召回率：",rc)
    print("F1:",(pr+rc)/(2*pr*rc))

solve()
print("W0:")
print(W[0])
print("W1:")
print(W[1])



