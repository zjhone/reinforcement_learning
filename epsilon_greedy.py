# Title: epsilon_greedy.py
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/06

import numpy as np
import matplotlib.pyplot as plt

class Gambling_machine:
    K = 0   # 0-摇臂
    R_MAX = 0  # 最大出币数
    probability = [[0]]
    # 摇臂赌博机构造函数
    def __init__(self, K_NUM = 2, R_MAX_NUM = 1, probability_list=[[0.5, 0.5], [0.5, 0.5]]):
        self.K = K_NUM
        self.R_MAX = R_MAX_NUM
        # self.R = np.random.randint(0, 3, size=k)
        self.probability = probability_list

    def show_para(self):
        print(f"摇臂数:{self.K}", f"奖赏函数：{self.probability}")

    def shake_handle(self, k):   # 执行摇臂动作
        ret = np.random.choice(self.R_MAX + 1, size=1, replace=True, p=self.probability[k])[0]
        return ret


def epsilon_greedy(GM, T, e):
    r = 0
    K = GM.K
    Q = np.zeros(K)
    cnt = np.zeros(K)
    Reward = [0]
    for t in range(T):
        if np.random.rand() < e:
            k = np.random.randint(K)   # 摇臂概率采用均匀分布
        else:
            k = Q.argmax()
        v = GM.shake_handle(k)
        r = r + v
        Q[k] = (Q[k]*cnt[k] + v)/(cnt[k] + 1)
        cnt[k] = cnt[k] + 1

        if t == 0:   # 更新平均累计奖赏
            Reward[t] = v
        else:
            Reward.append((Reward[t-1] * (t - 1) + v) / t)

    return r, cnt, Reward


if __name__ == "__main__":
    reward_probility_list = [[0.5, 0.3, 0.1, 0.1],  # 出币概率表
                             [0.1, 0.5, 0.2, 0.2],
                             [0.3, 0.6, 0.05, 0.05],
                             [0.8, 0.05, 0.05, 0.1],
                             [0.4, 0.2, 0.2, 0.2]]

    GM = Gambling_machine(5, 3, reward_probility_list)  # 初始化，默认是2摇臂。指定为5摇臂，每个臂最多出3个币

    GM.show_para()
    T = 3000   # 尝试次数
    # epsilon = 0.01  # 探索概率
    epsilon = [0.01, 0.1, 0.5, 0.8, 1]
    plt.figure()
    for i in epsilon:
        ret, cnt, Q_graph = epsilon_greedy(GM, T, i)
        print(f"epsilon={i}时最终收益：{ret}。摇臂详情：{cnt}。最后三次收益：{Q_graph[-4:-1]}")
        plt.plot(Q_graph, label="ta="+str(i))
    plt.legend(loc="best")
    plt.show()
