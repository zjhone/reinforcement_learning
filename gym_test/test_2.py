# Title: test_2.py
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/16


import gym
env = gym.make('MountainCarContinuous-v0') # try for different environments
observation = env.reset()
for t in range(10000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        if done:
            print("Finished after {} timesteps".format(t+1))
            break