# Title: sarsa.py   基于epsilon-greedy
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/04/01

import re
import gym
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 解决中文无法正常显示问题
fname = "/media/zjh/SDXC/linux-tools/font/simhei.ttf"
myfont = FontProperties(fname=fname)

class sarsa_algorithm:
    def __init__(self, grid_mdp):
        # 初始化模型参数E=<X, A, P, R>, 基于假设模型未知
        self.states = grid_mdp.getStates()
        self.actions = grid_mdp.getActions()
        self.state_and_action_space = grid_mdp.gett()
        self.terminate_states = grid_mdp.getTerminate_states()
        self.gamma = grid_mdp.getGamma()
        self.R = grid_mdp.getRewards()   # 奖赏
        self.pi = dict()
        self.pi_space = dict()
        self.probabilities = dict()
        self.QXA = dict()       # 记录状态-动作值
        self.counter = dict()   # 计数器
        # 引入的输入：
        self.x0 = 1  # 设置状态1为起始状态
        self.epsilon = 0.2
        self.T = 10  # 最大执行步数为8步

    # 初始化一些参数
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
        for state in self.pi_space.keys():
            self.pi[state] = self.pi_space[state][np.random.choice(len(self.pi_space[state]),
                                                                        size=1, replace=True)[0]]

        print('初始化的1个策略：', self.pi)

        # 将状态-动作函数置零
        for qstate in self.probabilities.keys():
            qqq = dict()
            for qaction in self.probabilities[qstate].keys():
                qqq[qaction] = 0.0
            self.QXA[qstate] = qqq

    # 返回奖赏及下一个状态
    def get_reward_and_next_state(self, state, action):
        key = "%d_%s" %(state, action)
        reward = self.R[key]
        next_state = self.state_and_action_space[key]
        return reward, next_state

    def get_action_based_epsilon_greedy(self, state):

        P = self.probabilities

        # 执行贪心策略获取下一个动作 a'
        if np.random.rand() < self.epsilon:  # epsilon的概率均匀选择
            temp = np.array(list(P[state].values()))  # 从起始位置x0出发，temp 代表当前状态下选取每个动作的概率值
            temp = temp / temp.sum()  # normalize归一化处理，保持总概率始终为1
            action = list(P[state].keys())[np.random.choice(len(P[state].keys()),
                                                     size=1, replace=True, p=temp)[0]]
        else:
            # 1-epsilon的概率选择当前最优动作（“摇臂”）, 判断依据是Q（x,a）
            action = max(self.QXA[state], key=self.QXA[state].get)

        return action


    # 核心算法：
    def sarsa_interate(self):

        MAX_NUM = 1000   # 最大迭代数量
        alpha = 0.1
        Qxa = self.QXA
        countxa = self.counter
        DELTA = []
        flags = 0
        for _ in range(MAX_NUM):
            # random的作用为变动采样起始点，避免轨迹为同一条
            self.x0 = np.random.choice(5, size=1, replace=True)[0] + 1
            x = self.x0
            a = self.pi[x]
            delta = 0.0
            old_q = 0.0

            # 通常T步内会到达终止点，避免算力浪费（died end） 或者采用条件语句跳出
            for t in range(self.T):

                r, x_ = self.get_reward_and_next_state(x, a)

                # 执行贪心策略获取下一个动作 a'
                a_ = self.get_action_based_epsilon_greedy(x_)

                # 对状态-动作值按照DP动态规划的思想进行更新
                old_q = Qxa[x][a]
                Qxa[x][a] = Qxa[x][a] + alpha * ( r + self.gamma * Qxa[x_][a_] - Qxa[x][a])
                delta = delta + abs(Qxa[x][a] - old_q)

                # 更新确定性策略pi，贪婪策略
                for xkt in Qxa.keys():
                    self.pi[xkt] = max(Qxa[xkt], key=Qxa[xkt].get)

                # 更新计数器
                if x in countxa :
                    if a in countxa[x]:
                        countxa[x][a] = countxa[x][a] + 1
                    else:
                        countxa[x][a] = 1
                else:
                    countxa[x] = dict()
                    countxa[x][a] = 1

                x = x_
                a = a_

                # 更新计数器
                if x in countxa :
                    if a in countxa[x]:
                        countxa[x][a] = countxa[x][a] + 1
                    else:
                        countxa[x][a] = 1
                else:
                    countxa[x] = dict()
                    countxa[x][a] = 1

                # 到达终止点，跳出子循环重来
                if x in self.terminate_states: break


            print(f'\033[0;33;40m此次最优状态-动作：{self.pi}\033[0m')

            DELTA.append(delta)
            condition = bool(delta < 1e-6)
            # 达到收敛条件，退出大循环
            if condition: break

        print(f'\n最终的状态-动作值函数： {Qxa}')
        print(f'计数器： {countxa}')
        print(f'\n\033[1;32;40m当前最佳策略：{self.pi}\033[0m')

        plt.figure()    # 绘制delta变化曲线
        plt.plot(DELTA)
        plt.rcParams['font.sans-serif'] = ['Ubuntu']  # 用来正常显示中文标签
        plt.xlabel('迭代次数', fontproperties=myfont)
        plt.ylabel('dalta', fontproperties=myfont)
        plt.show()


if __name__=="__main__":
    env = gym.make("GridWorld-v0")
    env.setState(1)
    # env.render()
    MDP = sarsa_algorithm(env)

    MDP.init_params()

    t0 = time.time()
    MDP.sarsa_interate()
    t1 = time.time()

    print("--------------------DONE--------------------")
    print(f"\n算法耗时： {t1-t0} s")
