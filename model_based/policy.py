# Title: policy.py  策略迭代算法
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/20

import gym
import time
import re
import numpy as np
# import gym.envs.classic_control.grid_mdp as grid_map


class policy_algorithm:
    def __init__(self, grid_mdp):
        # 初始化模型参数
        self.pi = dict()
        self.pi_space = dict()
        self.states = grid_mdp.getStates()
        self.actions = grid_mdp.getActions()
        self.v = [0.0 for i in range(len(self.states))]
        print("初始化状态函数V(s)：", self.v)
        self.state_and_action_space = grid_mdp.gett().keys()
        self.terminate_states = grid_mdp.terminate_states()
        self.gamma = grid_mdp.getGamma()

    # 策略选择函数，类似于K摇臂赌博机
    def pi_evaluate(self):
        for jt in self.state_and_action_space:  # 整理状态-动作空间
            temp = re.match(r'(.*)_(.*)', jt)
            if int(temp.group(1)) in self.pi_space.keys():
                self.pi_space[int(temp.group(1))].append(temp.group(2))
            else:
                li = []
                li.append(temp.group(2))
                self.pi_space[int(temp.group(1))] = li

        # for state in self.states:   # 随机初始化策略, 这个不够严谨，换用下面的for
        #     self.pi[state] = self.actions[np.random.choice(len(self.actions), size=1, replace=True)[0]]
        for state in self.pi_space.keys():
            self.pi[state] = self.pi_space[state][np.random.choice(len(self.pi_space[state]),
                                                                        size=1, replace=True)[0]]

    # 评估策略，返回状态值函数
    def policy_evaluate(self, grid_mdp):

        MAX_ITERATION = 100  #最大迭代次数

        for i in range(MAX_ITERATION):
            delta = 0.0
            for state in self.states:
                if state in self.terminate_states: continue
                # 更新参数
                action = self.pi[state]
                # flags代表时候到达重点，s代表下一个状态，r代表奖励
                flags, s, r = grid_mdp.transform(state, action)
                new_v = r + self.gamma * self.v[s-1]   # python列表是从0索引开始的，因此需要减1对齐
                # TODO 公式应当考虑pi求和，确定性策略更新的值函数不可用，求出的最优策略不可用
                delta = delta + abs(self.v[state-1] - new_v)
                self.v[state-1] = new_v
            if delta < 1e-3: break
            print(f'第{i}次迭代后', self.v)

        print("策略评估后V：", self.v)

    # 策略改进
    def policy_improve(self, grid_mdp):
        for state in self.states:
            if state in self.terminate_states: continue
            a1 = self.actions[0]
            # flags代表是否到达终点，s代表下一个状态，r代表奖励
            flags, s, r = grid_mdp.transform(state, a1)
            v1 = r + self.gamma * self.v[s-1]
            # 贪婪策略
            for action in self.actions:
                # flags代表时候到达重点，s代表下一个状态，r代表奖励
                flags, s, r = grid_mdp.transform(state, action)
                if v1 <= r + self.gamma * self.v[s-1]:
                    a1 = action
                    v1 = r + self.gamma * self.v[s-1]
            self.pi[state] = a1

    # 策略评估 + 策略改进 = 策略迭代算法
    def policy_iterate(self, grid_mdp):
        for i in range(1):
            self.policy_evaluate(grid_mdp)
            self.policy_improve(grid_mdp)
        print('最后的状态值：', self.v, '\n策略：', self.pi)


if __name__ == "__main__":
    env = gym.make("GridWorld-v0")
    env.reset()
    env.setState(1)
    env.render()
    time.sleep(2)

    DP = policy_algorithm(env)
    print('状态空间：', DP.states,type(DP.states[0]), '\n动作空间：', DP.actions, type(DP.actions[0]))
    DP.pi_evaluate()
    print('状态-动作空间：', DP.pi_space)
    print('初始化策略(确定性)：', DP.pi)

    # env.setState(7)  # 功能测试
    # print('当前状态：', env.getState())
    # env.render()
    # time.sleep(2)
    #
    # env.transform(7, 'w')
    # env.render()
    # print('当前状态：', env.getState())
    # time.sleep(5)

    print("-----CALCULATING-----")
    DP.policy_iterate(env)  # 策略迭代算法
    print("---------DONE--------")