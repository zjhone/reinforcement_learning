# Title: policy.py  策略迭代算法
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/20

import gym
import time
import re
import numpy as np
# import gym.envs.classic_control.grid_mdp as grid_map
#casfsfs

class policy_algorithm:
    def __init__(self, grid_mdp):
        self.v = [0, 0, 0, 0, 0, 0, 0, 0]
        self.pi = dict()
        self.pi_space = dict()
        self.states = grid_mdp.getStates()
        self.actions = grid_mdp.getActions()

    # 策略选择函数，类似于K摇臂赌博机
    def pi_evaluate(self, grid_mdp):
        for jt in grid_mdp.gett().keys():  # 整理状态-动作空间
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
        MAX_ITERATION = 1000
        for i in range(MAX_ITERATION):
            delta = 0.0
            for state in grid_mdp.states:
                if state in grid_mdp.terminate_states: continue
                # 更新参数
                action = self.pi[state]
                t, s, r = grid_mdp.transform(state, action)
                new_v = r + grid_mdp.gamma * self.v[s]
                delta = delta + abs(self.v[state] - new_v)
                self.v[state] = new_v
            if delta < 1e-6: break

    # 策略改进
    def policy_improve(self, grid_mdp):
        for state in grid_mdp.states:
            if state in grid_mdp.terminate_states: continue
            a1 = grid_mdp.actions[0]
            t, s, r = grid_mdp.transform(state, a1)
            v1 = r + grid_mdp.gamma * self.v[s]
            # 贪婪策略
            for action in grid_mdp.actions:
                t, s, r = grid_mdp.transform(state, action)
                if v1 < r + grid_mdp.gamma * self.v[s]:
                    a1 = action
                    v1 = r + grid_mdp.gamma * self.v[s]
            self.pi[state] = a1

    # 策略评估 + 策略改进 = 策略迭代算法
    def policy_iterate(self, grid_mdp):
        for i in range(100):
            self.policy_evaluate(grid_mdp)
            self.policy_improve(grid_mdp)


if __name__ == "__main__":
    env = gym.make("GridWorld-v0")
    env.reset()
    # env.render()

    DP = policy_algorithm(env)
    print(DP.states,type(DP.states[0]), DP.actions, type(DP.actions[0]))
    DP.pi_evaluate(env)
    print(DP.pi_space)
    print(DP.pi)

    print("-----DONE!-----")