# import re
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
    tmp = []
    for x in line:
        if(x != ''):
            tmp.append(float(x))
    arr.append(tmp)
arrtt = []
for i in range(400):
    tmp = []
    tmp.append(1.0)
    for j in range(13):
        tmp.append(arr[i][j])
    arrtt.append(tmp)
# 初始θ为1
theta = []
for i in range(0, 14):
    theta.append(1.0)
# 矩阵乘法


def mul(a, b):
    n = len(a)
    k = len(a[0])
    m = len(b[0])
    res = []
    for row in range(n):
        tmp = []
        for col in range(m):
            tt = 0.0
            for i in range(k):
                tt += a[row][i]*b[i][col]
            tmp.append(tt)
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
    ret = theta[0]
    for i in range(13):
        ret += theta[i+1]*x[i]
    return ret


def vector2matrix(a):
    res = []
    for x in a:
        res.append([x])
    return res


def matrix_sub(a, b):
    res = []
    for i in range(len(a)):
        tmp = []
        for j in range(len(a[i])):
            tmp.append(a[i][j]-b[i][j])
        res.append(tmp)
    return res


def matrix_T(a):
    res = []
    for i in range(len(a[0])):
        tmp = []
        for j in range(len(a)):
            tmp.append(a[j][i])
        res.append(tmp)
    return res


def matrix2vector(a):
    res = []
    for row in a:
        for x in row:
            res.append(x)
    return res


# start = time.perf_counter()
# 遍历前400个数据求θ,α为1
alpha = 0.00000715
m = 400
j = 1000000.0
# 损失值低于25就退出
arrtT = matrix_T(arrtt)
y = []
for i in range(400):
    y.append([arr[i][13]])

while(j >= 30):
    j = 0.0
    for i in range(0, 400):
        j += (h(arr[i])-arr[i][13])*(h(arr[i])-arr[i][13])
    j /= 2.0*400
    # print(j)
    mm = vector2matrix(theta)
    tt = matrix2vector(mul(arrtT, matrix_sub(mul(arrtt, mm), y)))
    theta = add(theta, mul_k(-(alpha/m), tt))


print("最后的theta下前400个样本的损失函数值：")
print(j)
print("theta取值：")
print(theta)

# 预测后面106个数据，计算损失值
j = 0.0
for i in range(400, 506):
    tmp = h(arr[i])-arr[i][13]
    j += tmp*tmp

j /= 2*106
print("后106个样本的损失函数值：")
print(j)
# end = time.perf_counter()
# print(end-start)
