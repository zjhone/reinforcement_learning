import numpy as np


# 求原列表的x折均值
def my_reshape(my_list, interval):
    ret = []
    tmp0 = []
    for i in range(len(my_list)):
        tmp0.append(my_list[i])
        if (i%interval == interval-1) or (i == len(my_list)-1):
            ret.append(tmp0)
            tmp0 = []
    SA = []
    tmp1 = 0
    for i in range(len(my_list)):
        SA.append(np.average(ret[tmp1]))
        if i%interval == interval-1:
           tmp1 = tmp1 + 1
    # print(ret)
    # print(SA)

    return SA


def cumulative(my_lsit):
    tmp = []
    for nt in my_lsit:
        tmp.append((np.sum(tmp)+nt)/(len(tmp)+1))
    # print(tmp)
    return tmp



def list_fit(my_list, num):
    x = list(i for i in range(len(my_list)))
    ret = np.polyval(np.polyfit(x, my_list, num), x)

    return ret

# a = [1,2,3,4,5,6]
# print(cumulative(a))
