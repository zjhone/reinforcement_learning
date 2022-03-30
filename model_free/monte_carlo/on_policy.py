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
        # 引入的输入：
        self.x0 = 1  # 设置状态1为起始状态
        self.epsilon = 0.5
        self.T = 10  # 最大执行步数为10步
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

    # 基于epsilon-greeedy的蒙特卡罗采样
    def trace_cal(self):
        P = self.probabilities   # 动作选择概率
        sampling_list = list()  # 采样是有顺序的，故不建议用字典来存储
        state = self.x0
        for t in range(self.T):  # 初始化T步轨迹
            temp = list(P[state].values())   # 从起始位置x0出发，temp 代表当前状态下选取每个动作的概率值
            action = list(P[state].keys())[np.random.choice(len(self.pi_space[state]),
                                                                        size=1, replace=True, p=temp)[0]]
            key = "%d_%s" % (state, action)
            r = self.R[key]
            sampling_list.append((state, action, r))
            state = self.state_and_action_space[key]
        # print("\n此次采样轨迹为： ", sampling_list)
        return sampling_list

    # 核心算法：
    def mente_carlo_interate(self):

        MAX_SAMPLING_NUM = 100   # 最大轨迹数量
        Qxa = dict()
        countxa = dict()

        for s in range(MAX_SAMPLING_NUM):
            sampling_temp = self.trace_cal()   # 采样
            for pt in sampling_temp:
                key = "%d_%s" %(pt[0], pt[1])
                if key in Qxa:
                    continue
                else:
                    Qxa[key] = 0.0
                    countxa[key] = 0

            for t in range(self.T):
                R = 0  # 求奖赏
                key = "%d_%s" %(sampling_temp[t][0], sampling_temp[t][1])
                for j in np.linspace(t+1, self.T-1, num=(self.T-t+1), dtype=int):
                    R = R + sampling_temp[t][2] / (self.T-t)

                Qxa[key] = (Qxa[key] * countxa[key] + R) / (countxa[key] + 1)
                countxa[key] = countxa[key] + 1

            # 对所有已见状态x更新策略pi
            for kt in Qxa.keys():
                temp = re.match(r'(.*)_(.*)', kt)
                temp_state = temp.group(1)
                temp_action = temp.group(2)


        print(Qxa)
        print((countxa))

if __name__=="__main__":
    env = gym.make("GridWorld-v0")
    env.setState(1)
    # env.render()
    MF = mento_carlo_on_policy(env)
    MF.init_probabilities()
    print(MF.pi_space)
    # MF.trace_cal()
    MF.mente_carlo_interate()
    # time.sleep(1)
    print("--------------------DONE--------------------")

