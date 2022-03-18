# Title: policy_iterate.py  策略迭代算法
# Author: Zhang Jianghu
# Date: 2022/03/12

import numpy as np
import gym
import time

if __name__ == "__main__":
    env = gym.make("GridWorld-v0")
    env.reset()
    env.render()
    time.sleep(1)
    print("Done!")
