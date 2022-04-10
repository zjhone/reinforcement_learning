# Title: value.py  值迭代算法-基于折扣
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/27

import numpy as np
import time
import gym
import re

class value_algorithm:
    def __init__(self, grid_mdp):
        # 初始化模型参数E=<X, A, P, R>
        self.states = grid_mdp.getStates()
        self.actions = grid_mdp.getActions()
        self.state_and_action_space = grid_mdp.gett()
        self.terminate_states = grid_mdp.getTerminate_states()
        self.gamma = grid_mdp.getGamma()
        self.pi = dict()
        self.new_pi = dict()
        self.pi_space = dict()
        self.probabilities = dict()
        self.v = [0.0 for i in range(len(self.states))]
        print("初始化状态函数V(s)：", self.v)

    # 策略选择函数，类似于K摇臂赌博机，同时更新概率函数
    def pi_evaluate(self):
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
    # 值迭代核心算法
    def value_iterate(self, grid_mdp):
        MAX_ITERATION = 100    # 累积奖赏参数
        for t in range(MAX_ITERATION):
            delta = 0.0
            for state in self.states:
                if state in self.terminate_states: continue
                a1 = self.pi_space[state][0]  # 每个状态以第一个动作为参考进行状态值（或状态-动作值）比较
                flag, next_state, r = grid_mdp.transform(state, a1)
                Qxa = r + self.gamma * self.v[next_state-1]  # 减1的操作是为了对齐，v为列表，python是从0 index开始的
                for action in self.pi_space[state]:
                    flag, next_state, r = grid_mdp.transform(state, action)
                    if Qxa < r + self.gamma * self.v[next_state-1]:
                        a1 = action
                        Qxa = r + self.gamma * self.v[next_state-1]
                v1 = Qxa   # 取最大值
                delta = delta + abs(v1 - self.v[state-1])
                self.pi[state] = a1
                self.v[state-1]  = v1

            if delta < 1e-6: break   # 设定收敛阈值
        print("返回的最优策略为： ", self.pi)

        # plt.figure()    # 绘制delta变化曲线
        # plt.plot(DELTA)
        # plt.rcParams['font.sans-serif'] = ['Ubuntu']  # 用来正常显示中文标签
        # plt.xlabel('策略评估迭代次数', fontproperties=myfont)
        # plt.ylabel('状态值函数变化差值dalta', fontproperties=myfont)
        # plt.show()

    def search_solution(self, query):
        '''
        :param query: 查询某个状态的最佳路径
        :return: 最佳状态-动作对（即最佳策略）
        '''
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



if __name__ =="__main__":
    env = gym.make("GridWorld-v1")
    # env.reset()
    env.setState(1)
    env.render()
    time.sleep(1)

    DP = value_algorithm(env)
    print('状态空间：', DP.states, type(DP.states[0]), '\n动作空间：', DP.actions, type(DP.actions[0]))
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
    DP.value_iterate(env)  # 策略迭代算法

    # 查询最优策略
    my_query = input("\033[0;32;40mPlease set the robot state:\033[0m")
    env.setState(int(my_query))
    env.render()
    time.sleep(1)

    print("\033[0;32;40mGUIDING……\033[0m")
    best_road = DP.search_solution(int(my_query))
    time.sleep(1)
    env.guide(best_road)
    time.sleep(2)

    print("--------------------DONE--------------------")