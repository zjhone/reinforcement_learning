# Title: test_3.py
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/4/10

import gym
import time

if __name__=="__main__":
    env = gym.make("GridWorld-v1")
    env.reset()
    env.render()
    time.sleep(10)
