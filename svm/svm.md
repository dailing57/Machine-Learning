首先把x,y结合起来进行shuffle

然后再分别取x和y

把数据分成10份

使用svm（rbf为核函数，惩罚系数为默认的1）算法跑10轮，对于第 i 轮，取 i 第轮，取第 i 个x，y作为测试样本，其余的是训练样本

最后对十轮的精确值、正准确率、正召回率、负准确率、负召回率取平均。

precision:  0.8875
positive precision:  0.7993244785594922
positive recall:  0.9380583722898301
negative precision:  0.8888199375219836
negative recall 0.6588475794358147

