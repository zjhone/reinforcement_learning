import re
import logging
import numpy
import time
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)


# 求原列表的x折
def my_reshape(my_list, interval):
    ret = []
    tmp0 = []
    for i in range(len(my_list)):
        tmp0.append(my_list[i])
        if (i%interval == interval-1) or (i == len(my_list)-1):
            ret.append(tmp0)
            tmp0 = []
    # print(ret)
    return ret

# 二维列表索引，返回匹配的第一个值的索引，找不到的话返回None
def get_index_from_two_dim_list(L, value):
    for i in range(len(L)):
        for j in range(len(L[0])):
            if L[i][j] == value:
                return i, j
    return 0,0


class MatrixEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, row=8, column=8):
        # row,colum = map(int,input('输入row, column空格隔开:\n').split())
        self.size = (self.width, self.height) = (40, 40)  # 单个框的大小
        self.row = row
        self.column = column
        self.states = [i + 1 for i in range(row * column)]   # #状态空间

        # self.states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,] #状态空间
        # self.x=[150,250,350,450,150,250,350,450,150,250,350,450,150,250,350,450,]
        # self.y=[450,450,450,450,350,350,350,350,250,250,250,250,150,150,150,150,]
        self.xpadding = 1.5 * self.width
        self.ypadding = 1.5 * self.height
        self.x = [self.xpadding+self.width*(i) for j in range(row) for i in range(column)]
        self.y = [((row+2)*self.height-self.ypadding)-self.height*(j) for j in range(row) for i in range(column)]

        # print(self.x)
        # print(self.y)

        self.destinations = [row*column]
        self.traps = [(column*(row-1)+1), column]

        self.terminate_states = {self.traps[0]:1, self.traps[1]:1, self.destinations[0]:1}   # 终止状态为字典格式

        self.actions = ['n','e','s','w']

        self.rewards = dict();        #回报的数据结构为字典
        STATE_TABLE = my_reshape(self.states, column)
        print(self.column, 'x', self.row)
        for r in range(row):
            for c in range(column):
                if STATE_TABLE[r][c] in self.terminate_states:  # 让终止节点指向自己
                    self.rewards[str(STATE_TABLE[r][c]) + '_n'] = -1.0
                    continue
                else:
                    self.rewards[str(STATE_TABLE[r][c]) + '_e'] = -1.0
                    self.rewards[str(STATE_TABLE[r][c]) + '_w'] = -1.0
                    self.rewards[str(STATE_TABLE[r][c]) + '_s'] = -1.0
                    self.rewards[str(STATE_TABLE[r][c]) + '_n'] = -1.0

                    if STATE_TABLE[r][c] == r*column+1:
                        self.rewards.pop(str(STATE_TABLE[r][c]) + '_w')
                    if STATE_TABLE[r][c] == (r+1)*column:
                        self.rewards.pop(str(STATE_TABLE[r][c]) + '_e')
                    if STATE_TABLE[r][c]<= 1*column:
                        self.rewards.pop(str(STATE_TABLE[r][c]) + '_n')
                    if STATE_TABLE[r][c] > (row-1)*column:
                        self.rewards.pop(str(STATE_TABLE[r][c]) + '_s')

        # print(len(self.rewards), dict(sorted(self.rewards.items())).keys())
        # self.rewards=dict()
        # self.rewards['1_s'] = -1.0
        # self.rewards['1_e'] = -1.0
        # self.rewards['2_w'] = -1.0
        # self.rewards['2_e'] = -1.0
        # self.rewards['2_s'] = -1.0
        # self.rewards['3_w'] = -1.0
        # self.rewards['3_e'] = -10.0
        # self.rewards['3_s'] = -1.0
        #
        # self.rewards['4_n'] = -1.0
        #
        # self.rewards['5_n'] = -1.0
        # self.rewards['5_e'] = -1.0
        # self.rewards['5_s'] = -1.0
        # self.rewards['6_n'] = -1.0
        # self.rewards['6_w'] = -1.0
        # self.rewards['6_e'] = -1.0
        # self.rewards['6_s'] = -1.0
        # self.rewards['7_n'] = -1.0
        # self.rewards['7_w'] = -1.0
        # self.rewards['7_e'] = -1.0
        # self.rewards['7_s'] = -1.0
        # self.rewards['8_n'] = -10.0
        # self.rewards['8_w'] = -1.0
        # self.rewards['8_s'] = -1.0
        # self.rewards['9_n'] = -1.0
        # self.rewards['9_e'] = -1.0
        # self.rewards['9_s'] = -10.0
        # self.rewards['10_n'] = -1.0
        # self.rewards['10_w'] = -1.0
        # self.rewards['10_e'] = -1.0
        # self.rewards['10_s'] = -1.0
        # self.rewards['11_n'] = -1.0
        # self.rewards['11_w'] = -1.0
        # self.rewards['11_e'] = -1.0
        # self.rewards['11_s'] = -1.0
        # self.rewards['12_n'] = -1.0
        # self.rewards['12_w'] = -1.0
        # self.rewards['12_s'] = 10.0
        #
        # self.rewards['13_n'] = -1.0
        #
        # self.rewards['14_n'] = -1.0
        # self.rewards['14_w'] = -10.0
        # self.rewards['14_e'] = -1.0
        # self.rewards['15_n'] = -1.0
        # self.rewards['15_w'] = -1.0
        # self.rewards['15_e'] = 10.0
        #
        # self.rewards['16_n'] = -1.0
        #
        # print(len(self.rewards), dict(sorted(self.rewards.items())).keys())
        # print('\n\n')

        self.t = dict();    #状态转移的数据格式为字典
        for sa in self.rewards.keys():
            temp = re.match(r'(.*)_(.*)', sa)
            if int(temp.group(1)) in self.terminate_states:
                self.t[sa] = int(temp.group(1))
            else:
                r, c = get_index_from_two_dim_list(STATE_TABLE, int(temp.group(1)))
                if temp.group(2) == 'n':
                    r = r - 1
                elif temp.group(2) == 's':
                    r = r + 1
                elif temp.group(2) == 'e':
                    c = c + 1
                elif temp.group(2) == 'w':
                    c = c - 1
                self.t[sa] = STATE_TABLE[r][c]

        # print(len(self.t), dict(sorted(self.t.items())))
        #
        # self.t = dict();             #状态转移的数据格式为字典
        # self.t['1_s'] = 5
        # self.t['1_e'] = 2
        # self.t['2_w'] = 1
        # self.t['2_e'] = 3
        # self.t['2_s'] = 6
        # self.t['3_w'] = 2
        # self.t['3_e'] = 4
        # self.t['3_s'] = 7
        #
        # self.t['4_n'] = 4
        #
        # self.t['5_n'] = 1
        # self.t['5_e'] = 6
        # self.t['5_s'] = 9
        # self.t['6_n'] = 2
        # self.t['6_w'] = 5
        # self.t['6_e'] = 7
        # self.t['6_s'] = 10
        # self.t['7_n'] = 3
        # self.t['7_w'] = 6
        # self.t['7_e'] = 8
        # self.t['7_s'] = 11
        # self.t['8_n'] = 4
        # self.t['8_w'] = 7
        # self.t['8_s'] = 12
        # self.t['9_n'] = 5
        # self.t['9_e'] = 10
        # self.t['9_s'] = 13
        # self.t['10_n'] = 6
        # self.t['10_w'] = 9
        # self.t['10_e'] = 11
        # self.t['10_s'] = 14
        # self.t['11_n'] = 7
        # self.t['11_w'] = 10
        # self.t['11_e'] = 12
        # self.t['11_s'] = 15
        # self.t['12_n'] = 8
        # self.t['12_w'] = 11
        # self.t['12_s'] = 16
        #
        # self.t['13_n'] = 13
        #
        # self.t['14_n'] = 10
        # self.t['14_w'] = 13
        # self.t['14_e'] = 15
        # self.t['15_n'] = 11
        # self.t['15_w'] = 14
        # self.t['15_e'] = 16
        #
        # self.t['16_n'] = 16
        #
        # print(len(self.t), dict(sorted(self.t.items())))
        # print('\n\n\n')

        for rd in self.rewards.keys():
            if self.t[rd] in self.traps:
                self.rewards[rd] = -100
            elif self.t[rd] in self.destinations:
                self.rewards[rd] = 100
            else:
                continue

        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getState(self):
        return self.state

    def getActions(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def gett(self):
        return self.t

    def getRewards(self):
        return self.rewards

    def setState(self,s):
        self.state=s
        print("当前状态被设置为：", self.state)

    def transform(self, state, action):
        '''
        :param state: 当前状态
        :param action: 执行的动作
        :return: is_terminal代表是否到达终点，next_state代表下一个状态，r代表奖励
        '''
        is_terminal = False
        key = "%d_%s"%(state, action)   #将状态和动作组成字典的键值

        if state in self.terminate_states:
            is_terminal = True
            next_state = state
            r = 0.0
            return is_terminal, next_state, r

        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state

        self.state = next_state

        if next_state in self.terminate_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]

        return is_terminal, next_state, r

    # 路径导航
    def guide(self, road_list):
        if road_list:
            for pt in road_list:
                state_and_action = "%d_%s"%(pt[0], pt[1])   #将状态和动作组成字典的键值
                self.state = self.t[state_and_action]
                time.sleep(1)
                self.render()
            print("\033[0;32;40mGUIDE COMPELETED!\033[0m")
        else:
            print("\033[0;31;40mROAD LIST EMPTY!\033[0m")


    def _step(self, action):
        #系统当前状态
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        key = "%d_%s"%(state, action)   #将状态和动作组成字典的键值

        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]


        return next_state, r,is_terminal,{}

    def _reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        # 窗口大小
        screen_width = (self.column + 2) * self.width
        screen_height = (self.row + 2) * self.width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #创建网格世界

            point_list = list()
            for hori in range(self.row+1):   # 增加坐标点
                point_list.append(((1*self.width, (hori+1)*self.width),((self.column+1)*self.width, (hori+1)*self.width)))
            for vert in range(self.column+1):
                point_list.append((((vert+1)*self.height, (self.row+1)*self.height), ((vert+1)*self.height, 1*self.height)))

            self.line = list()
            for point in point_list:
                self.line.append(rendering.Line(point[0], point[1]))

            for i in range(len(self.line)):
                self.line[i].set_color(0, 0, 0)

            # self.line1 = rendering.Line((100, 100), (500, 100))
            # self.line2 = rendering.Line((100, 200), (500, 200))
            # self.line3 = rendering.Line((100, 300), (500, 300))
            # self.line4 = rendering.Line((100, 400), (500, 400))
            # self.line5 = rendering.Line((100, 500), (500, 500))
            #
            # self.line6 = rendering.Line((100, 500), (100, 100))
            # self.line7 = rendering.Line((200, 500), (200, 100))
            # self.line8 = rendering.Line((300, 500), (300, 100))
            # self.line9 = rendering.Line((400, 500), (400, 100))
            # self.line10= rendering.Line((500, 500), (500, 100))


            #创建第一个骷髅
            self.kulo1 = rendering.make_circle(0.5*self.width)
            self.circletrans = rendering.Transform(translation=(self.xpadding, self.ypadding))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(1,0,0)
            #创建第二个骷髅
            self.kulo2 = rendering.make_circle(0.5*self.width)
            self.circletrans = rendering.Transform(translation=(self.xpadding + (self.column-1)*self.width,
                                                                self.ypadding + (self.row-1)*self.height))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(1, 0, 0)
            #创建金条
            self.gold = rendering.make_circle(0.5*self.width)
            self.circletrans = rendering.Transform(translation=(self.xpadding + (self.column-1)*self.width,
                                                                self.ypadding))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(0.2, 0.9, 0.2)
            #创建机器人
            self.robot= rendering.make_circle(0.3*self.width)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0, 0, 0)

            # self.line1.set_color(0, 0, 0)
            # self.line2.set_color(0, 0, 0)
            # self.line3.set_color(0, 0, 0)
            # self.line4.set_color(0, 0, 0)
            # self.line5.set_color(0, 0, 0)
            # self.line6.set_color(0, 0, 0)
            # self.line7.set_color(0, 0, 0)
            # self.line8.set_color(0, 0, 0)
            # self.line9.set_color(0, 0, 0)
            # self.line10.set_color(0, 0, 0)

            for i in range(len(self.line)):
                self.viewer.add_geom(self.line[i])

            # self.viewer.add_geom(self.line1)
            # self.viewer.add_geom(self.line2)
            # self.viewer.add_geom(self.line3)
            # self.viewer.add_geom(self.line4)
            # self.viewer.add_geom(self.line5)
            # self.viewer.add_geom(self.line6)
            # self.viewer.add_geom(self.line7)
            # self.viewer.add_geom(self.line8)
            # self.viewer.add_geom(self.line9)
            # self.viewer.add_geom(self.line10)

            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        # 状态转移
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])

        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
