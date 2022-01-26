import numpy as np
from sklearn import svm

ex = np.loadtxt('ex4x.dat')
ey = np.loadtxt('ex4y.dat')
TOTAL_NUM = 80

tmp = np.insert(ex,0,values=ey,axis=1)
np.random.shuffle(tmp)
ey = tmp[:,0]
ex = tmp[:,1:]

K = 10
GROUP_NUM = int(TOTAL_NUM/K)# 每一组K个数据，总共这么多个组
clf = svm.SVC()

precision = 0
TP = 0
TN = 0
FP = 0
FN = 0
positive_precision = 0
positive_recall = 0
negative_precision = 0
negative_recall = 0
PP = 0
PR = 0
NP = 0
NR = 0
for j in range(K):# K个数据中第j个作为测试数据
    cnt = 0
    for i in range(TOTAL_NUM):
        baseId = int(i/K)
        train_x_tmp = []
        train_y_tmp = []
        for k in range(K):
            if(k!=j):
                train_x_tmp.append(ex[baseId+k])
                train_y_tmp.append(ey[baseId+k])
        if i % K == 0:
            clf.fit(train_x_tmp,train_y_tmp)
            pre = clf.predict([ex[baseId+j]])
            if(pre[0] == ey[baseId+j]):
                cnt += 1
            if(pre[0] == 0):
                if( ey[baseId+j] == 0):
                    TN += 1
                else:
                    FN += 1
            else:
                if( ey[baseId+j] == 1):
                    TP += 1
                else:
                    FP += 1
    if(TP+FP!=0):
        PP+=1
        positive_precision += TP / (TP + FP)
    if(TP + FN!=0):
        PR+=1
        positive_recall += TP / (TP + FN)
    if(TN+FN!=0):
        NP+=1
        negative_precision += TN/(TN+FN)
    if(TN+FP!=0):
        NR+=1
        negative_recall += TN/(TN+FP)
    precision += cnt/GROUP_NUM# 所有组正确的数量/组的数量

print("precision: ",precision/K)
print("positive precision: ",positive_precision/PP)
print("positive recall: ",positive_recall/PR)
print("negative precision: ",negative_precision/NP)
print("negative recall",negative_recall/NR)


