# Title: off_policy.py  基于epsilon-greedy
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/28

import re
import os
import gym
import time
import copy
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
        self.QXA = dict()       # 记录状态-动作值
        self.counter = dict()   # 计数器
        # 引入的输入：
        self.x0 = 1  # 设置状态1为起始状态
        self.epsilon = 0.2
        self.T = 10  # 最大执行步数为8步
        self.v = [0.0 for i in range(len(self.states))]

        print("初始化状态函数V(s)：", self.v)

    # 初始化概率函数
    def init_params(self):
        for jt in self.state_and_action_space.keys():  # 整理状态-动作空间
            temp = re.match(r'(.*)_(.*)', jt)
            if int(temp.group(1)) in self.pi_space.keys():
                self.pi_space[int(temp.group(1))].append(temp.group(2))
            else:
                li = []
                li.append(temp.group(2))
                self.pi_space[int(temp.group(1))] = li

        print('状态-动作空间：', self.pi_space)

        for pt in self.pi_space.keys():  # 初始化状态动作概率
            paction = dict()
            for jt in self.pi_space[pt]:
                paction[jt] = 1 / len(self.pi_space[pt])  # 每个状态均匀分布
            self.probabilities[pt] = paction

        print('状态-动作概率pi(x,a): ', self.probabilities)

        # 按照均匀随机策略（抽象不具体）初始化一个策略（具体确定）
        for state in self.pi_space.keys():  # 初始化策略（均匀分布）
            self.pi[state] = self.pi_space[state][np.random.choice(len(self.pi_space[state]),
                                                                        size=1, replace=True)[0]]

        print('初始化的1个策略：', self.pi)

        # 将状态-动作函数置零
        for qstate in self.probabilities.keys():
            qqq = dict()
            for qaction in self.probabilities[qstate].keys():
                qqq[qaction] = 0.0
            self.QXA[qstate] = qqq

    # 基于epsilon-greeedy的蒙特卡罗采样
    def trace_cal(self):

        smp_list = list()  # 采样是有顺序的，故不建议用字典来存储
        state = self.x0
        P = self.probabilities

        for t in range(self.T):  # 初始化T步轨迹
            if np.random.rand() < self.epsilon:  # epsilon的概率均匀选择
                temp = np.array(list(P[state].values()))  # 从起始位置x0出发，temp 代表当前状态下选取每个动作的概率值
                temp = temp/temp.sum()   # normalize归一化处理，保持总概率始终为1
                action = list(P[state].keys())[np.random.choice(len(P[state].keys()),
                                                                size=1, replace=True, p=temp)[0]]
            else:
                # 1-epsilon的概率选择当前最优动作（“摇臂”）, 判断依据是Q（x,a）
                action = max(self.QXA[state], key=self.QXA[state].get)

            key = "%d_%s" % (state, action)
            r = self.R[key]
            smp_list.append((state, action, r))  # 采用列表+元组
            state = self.state_and_action_space[key]

        return smp_list

    # 核心算法：
    def mente_carlo_interate(self):

        MAX_SAMPLING_NUM = 1000   # 最大轨迹数量
        Qxa = self.QXA
        countxa = self.counter
        P_AXU = list()   # 修正概率

        for s in range(MAX_SAMPLING_NUM):

            # random的作用为变动采样起始点，避免采到的轨迹为同一条
            self.x0 = np.random.choice(5, size=1, replace=True)[0] + 1
            smp = self.trace_cal()   # 原始策略下epsilon-greedy采样
            print(f"\n第\033[1;32;40m{s}\033[0m次采样轨迹为： {smp}")

            # 更新修正概率：
            for pt in smp:
                if pt[1] == self.pi[pt[0]]:
                    P_AXU.append(1-self.epsilon+(self.epsilon/len(self.probabilities[pt[0]])))
                else:
                    P_AXU.append(self.epsilon/len(self.probabilities[pt[0]]))
            # print('修正概率: ', P_AXU)

            for pt in smp:   # 此循环用来计数
                if pt[0] in countxa :
                    if pt[1] in countxa[pt[0]]:
                        continue
                    else:
                        countxa[pt[0]][pt[1]] = 0
                else:
                    countxa[pt[0]] = dict()
                    countxa[pt[0]][pt[1]] = 0

            for t in range(self.T):
                R = 0.0  # 求奖赏
                p_axu = 1.0
                for i in np.linspace(t, self.T-1, num=self.T, dtype=int):
                    R = R + (smp[i][2] / (self.T-t))  # 未修正
                    if smp[i][1] == self.pi[smp[i][0]]:
                        p_axu = p_axu * P_AXU[i]
                R = R * p_axu   # 按照公式修正

                # 值函数更新
                Qxa[smp[t][0]][smp[t][1]] = (Qxa[smp[t][0]][smp[t][1]] *
                                                                 countxa[smp[t][0]][smp[t][1]] + R) / \
                                                                (countxa[smp[t][0]][smp[t][1]] + 1)

                # 更新计数器
                countxa[smp[t][0]][smp[t][1]] = countxa[smp[t][0]][smp[t][1]] + 1

            # 对所有已见状态x
            for xkt in Qxa.keys():
                # new_pi 代表当前最优动作
                self.new_pi[xkt] = max(Qxa[xkt], key=Qxa[xkt].get)

            print(f'\033[0;33;40m此次最优状态-动作：{self.new_pi}\033[0m')

            # 更新确定性策略pi以及pi(x,a)
            self.pi = self.new_pi
            self.QXA = Qxa

        print(f'\n最终的状态-动作值函数： {Qxa}')
        print(f'计数器： {countxa}')
        print(f'\n\033[1;32;40m当前最佳策略：{self.pi}\033[0m')


if __name__=="__main__":
    env = gym.make("GridWorld-v0")
    env.setState(1)
    # env.render()
    MF = mento_carlo_on_policy(env)

    MF.init_params()

    t0 = time.time()
    MF.mente_carlo_interate()
    t1 = time.time()

    print("--------------------DONE--------------------")
    print(f"\n算法耗时： {t1-t0} s")

