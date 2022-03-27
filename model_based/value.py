# Title: value.py  值迭代算法-基于折扣
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/27

import numpy as np
import time
import gym


class value_algorithm:
    def __init__(self, grid_mdp):
        self.v = []

