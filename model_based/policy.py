# Title: policy.py  策略迭代算法
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/20

import gym
import numpy as np
# import gym.envs.classic_control.grid_mdp as grid_map



class policy_algorithm:
    def __init__(self, grid_mdp):
        self.state_and_action_space = grid_mdp.gett()
        self.v = [0, 0, 0, 0, 0, 0, 0, 0]
        self.pi = {}

    def pick_action(self, pi):
        new_pi = dict()


        return new_pi

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
                delta = abs(self.v[state] - new_v)
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
