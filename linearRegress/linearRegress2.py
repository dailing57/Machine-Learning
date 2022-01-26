#import re
# import time
# 读入数据
inputfile = open('housing.data')
line = inputfile.readline()
arrPre = []
while line:
    i = 0
    tt = []
    while(i < len(line)):
        tmp = ""
        while(i < len(line) and line[i] != ' ' and line[i] != '\n'):
            tmp += line[i]
            i += 1
        i += 1
        tt.append(tmp)
    arrPre.append(tt)
    #arrPre.append(re.findall('\d*\.?\d*', line))
    line = inputfile.readline()
arr = []
for line in arrPre:
    tmp = [1.0]
    for x in line:
        if(x != ''):
            tmp.append(float(x))
    arr.append(tmp)
# 初始θ为1
theta = []
for i in range(0, 14):
    theta.append(1.0)
# 矩阵乘法


def mul(a, b):
    n = len(a)
    k = len(a[0])
    m = len(b)
    res = []
    for row in range(n):
        for col in range(m):
            tmp = []
            for i in range(k):
                tmp.append(a[row][k]*b[k][col])
        res.append(tmp)
    return res
# 标量乘向量


def mul_k(k, a):
    res = []
    for x in a:
        res.append(x*k)
    return res
# 向量加


def add(a, b):
    res = []
    for i in range(len(a)):
        res.append(a[i]+b[i])
    return res


# h函数
def h(x):
    ret = 0.0
    for i in range(14):
        ret += theta[i]*x[i]
    return ret


# start = time.perf_counter()
# 遍历前400个数据求θ,α为1
alpha = 0.00000715
m = 400
j = 100000.0
# 损失值低于20就退出
while(j >= 30):
    j = 0.0
    for i in range(0, 400):
        j += (h(arr[i])-arr[i][14])*(h(arr[i])-arr[i][14])
    j /= 2.0*400
    sum = [0.0 for i in range(14)]
    for i in range(0, 400):
        tt = (h(arr[i])-arr[i][14])
        sum = add(sum, mul_k(tt, arr[i]))
    theta = add(theta, mul_k(-(alpha/m), sum))
    # print(j)
print("最后的theta下前400个样本的损失函数值：")
print(j)
print("theta取值：")
print(theta)

# 预测后面106个数据，计算损失值
j = 0.0
for i in range(400, 506):
    tmp = h(arr[i])-arr[i][14]
    j += tmp*tmp

j /= 2*106
print("后106个样本的损失函数值：")
print(j)
# end = time.perf_counter()
# print(end-start)
