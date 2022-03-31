# Title: on_policy.py  基于epsilon-greedy
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/28

import re
import gym
import time
import numpy as np

class mento_carlo_on_policy:
    def __init__(self, grid_mdp):
        # 初始化模型参数E=<X, A, P, R>, 基于假设模型未知
        self.states = grid_mdp.getStates()
        self.actions = grid_mdp.getActions()
        self.state_and_action_space = grid_mdp.gett()
        self.terminate_states = grid_mdp.getTerminate_states()
        self.gamma = grid_mdp.getGamma()
        self.R = grid_mdp.getRewards()   # 奖赏
        self.pi = dict()
        self.new_pi = dict()
        self.pi_space = dict()
        self.probabilities = dict()
        self.new_probabilities = dict()
        # 引入的输入：
        self.x0 = 1  # 设置状态1为起始状态
        self.epsilon = 0.2
        self.T = 8  # 最大执行步数为8步
        self.v = [0.0 for i in range(len(self.states))]
        # self.Qxa = dict()  # 状态-动作值函数
        print("初始化状态函数V(s)：", self.v)

    # 初始化概率函数
    def init_probabilities(self):
        for jt in self.state_and_action_space.keys():  # 整理状态-动作空间
            temp = re.match(r'(.*)_(.*)', jt)
            if int(temp.group(1)) in self.pi_space.keys():
                self.pi_space[int(temp.group(1))].append(temp.group(2))
            else:
                li = []
                li.append(temp.group(2))
                self.pi_space[int(temp.group(1))] = li

        for pt in self.pi_space.keys():  # 初始化状态动作概率
            paction = dict()
            for jt in self.pi_space[pt]:
                paction[jt] = 1 / len(self.pi_space[pt])  # 每个状态均匀分布
            self.probabilities[pt] = paction

        print('状态-动作概率pi(x,a): ', self.probabilities)

        for state in self.pi_space.keys():  # 初始化策略（均匀分布）
            self.pi[state] = self.pi_space[state][np.random.choice(len(self.pi_space[state]),
                                                                        size=1, replace=True)[0]]

    # 基于epsilon-greeedy的蒙特卡罗采样
    def trace_cal(self):
        P = self.probabilities   # 动作选择概率
        smp_list = list()  # 采样是有顺序的，故不建议用字典来存储
        state = self.x0
        for t in range(self.T):  # 初始化T步轨迹
            temp = list(P[state].values())   # 从起始位置x0出发，temp 代表当前状态下选取每个动作的概率值
            action = list(P[state].keys())[np.random.choice(len(self.pi_space[state]),
                                                                        size=1, replace=True, p=temp)[0]]
            key = "%d_%s" % (state, action)
            r = self.R[key]
            smp_list.append((state, action, r))
            state = self.state_and_action_space[key]

        return smp_list

    # 核心算法：
    def mente_carlo_interate(self):

        MAX_SAMPLING_NUM = 10000   # 最大轨迹数量
        Qxa = dict()
        countxa = dict()

        for s in range(MAX_SAMPLING_NUM):
            smp = self.trace_cal()   # 采样
            self.x0 = np.random.choice(5, size=1, replace=True)[0]+1 # 变动采样起始点
            print(f"\n第{s}次采样轨迹为： {smp}")
            for pt in smp:
                # 拓展Qxa与countxa
                if pt[0] in Qxa :
                    if pt[1] in Qxa[pt[0]]:
                        continue
                    else:
                        Qxa[pt[0]][pt[1]] = 0.0
                        countxa[pt[0]][pt[1]] = 0
                else:
                    Qxa[pt[0]] = dict()
                    countxa[pt[0]] = dict()
                    Qxa[pt[0]][pt[1]] = 0.0
                    countxa[pt[0]][pt[1]] = 0

            for t in range(self.T):
                R = 0  # 求奖赏
                for j in np.linspace(t+1, self.T-1, num=(self.T-t+1), dtype=int):
                    R = R + (smp[t][2] / (self.T-t))
                # 值函数更新
                Qxa[smp[t][0]][smp[t][1]] = (Qxa[smp[t][0]][smp[t][1]] *
                                                                 countxa[smp[t][0]][smp[t][1]] + R) / \
                                                                (countxa[smp[t][0]][smp[t][1]] + 1)
                # 计数器
                countxa[smp[t][0]][smp[t][1]] = countxa[smp[t][0]][smp[t][1]] + 1

            # 对所有已见状态x
            for xkt in Qxa.keys():
                # new_pi 代表当前最优动作
                self.new_pi[xkt] = max(Qxa[xkt], key=Qxa[xkt].get)

            print('此次最优状态-动作：', self.new_pi)

            # 更新确定性策略pi以及pi(x,a)
            for state in self.pi.keys():
                # epislon-greedy算法
                P = self.probabilities  # 动作选择概率(均匀)
                # 以epsilon的概率按照均匀概率选择动作
                if np.random.rand() < self.epsilon or state not in self.new_pi:
                    temp = list(P[state].values())
                    self.pi[state] = list(P[state].keys())[np.random.choice(len(self.pi_space[state]),
                                                                        size=1, replace=True, p=temp)[0]]
                else:  # 以1-epsilon的概率取当前最佳摇臂
                    self.pi[state] = self.new_pi[state]

            print(Qxa)
            print(countxa)
        print(f'\n\033[0;32;40m当前最佳策略：{self.pi}\033[0m')


if __name__=="__main__":
    env = gym.make("GridWorld-v0")
    env.setState(1)
    # env.render()
    MF = mento_carlo_on_policy(env)
    MF.init_probabilities()
    print('动作空间细节：', MF.pi_space)
    t0 = time.time()
    MF.mente_carlo_interate()
    t1 = time.time()
    print(f"\n算法耗时： {t1-t0} s")
    print("--------------------DONE--------------------")

