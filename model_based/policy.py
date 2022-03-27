# Title: policy.py  策略迭代算法
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/20

import gym
import time
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 解决中文无法显示问题
fname = "/media/zjh/SDXC/linux-tools/linux_font/simhei.ttf"
myfont = FontProperties(fname=fname)
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
        self.state_and_action_space = grid_mdp.gett()
        self.terminate_states = grid_mdp.getTerminate_states()
        self.gamma = grid_mdp.getGamma()

    # 策略选择函数，类似于K摇臂赌博机
    def pi_evaluate(self):
        for jt in self.state_and_action_space.keys():  # 整理状态-动作空间
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

        MAX_ITERATION = 1000  #最大迭代次数
        DELTA = []
        for i in range(MAX_ITERATION):
            delta = 0.0
            for state in self.states:
                if state in self.terminate_states: continue
                CNT = len(self.pi_space[state])
                new_v = 0.0
                for action in self.pi_space[state]:
                    flags, s, r = grid_mdp.transform(state, action)
                    new_v = new_v + (r + self.gamma * self.v[s - 1]) / CNT

                delta = delta + abs(self.v[state-1] - new_v)
                self.v[state-1] = new_v
            DELTA.append(delta)
            if delta < 1e-7: break
            print(f'第{i}次策略评估：', self.v)

        # plt.figure()    # 绘制delta变化曲线
        # plt.plot(DELTA)
        # plt.rcParams['font.sans-serif'] = ['Ubuntu']  # 用来正常显示中文标签
        # plt.xlabel('策略评估迭代次数', fontproperties=myfont)
        # plt.ylabel('状态值函数变化差值dalta', fontproperties=myfont)
        # plt.show()

    # 策略改进
    def policy_improve(self, grid_mdp):
        for state in self.states:
            if state in self.terminate_states: continue
            a1 = self.pi_space[state][0]
            # TODO 这里也有一点问题，为何要取action[0]?
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
        print('最后的状态值：', self.v, '\n均匀随机性策略（学习结果）：', self.pi)

    def search_solution(self, query):
        ret = list()
        for i in range(100):  # 路径查询100次
            if query in self.terminate_states:
                # print("结束!")
                break
            else:
                action = self.pi[query]
                key = "%d_%s"%(query, action)
                # print(f'{query}-->{action}')
                ret.append((query, action))
                query = self.state_and_action_space[key]
                continue
            print("\033[0;31;40m不存在最好路径!\033[0m")

        print(f"\033[0;32;40m最佳路径： {ret}\033[0m")
        return ret



if __name__ == "__main__":
    env = gym.make("GridWorld-v0")
    # env.reset()
    env.setState(1)
    env.render()
    time.sleep(1)

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

    print("--------------------CALCULATING--------------------")
    DP.policy_iterate(env)  # 策略迭代算法

    my_query = input("\033[0;32;40mPlease set the robot state:\033[0m")
    env.setState(int(my_query))
    env.render()
    time.sleep(1)

    print("\033[0;32;40mGUIDING……\033[0m")
    best_road = DP.search_solution(int(my_query))
    time.sleep(2)
    env.guide(best_road)
    time.sleep(5)
    print("--------------------DONE--------------------")