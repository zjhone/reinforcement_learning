# Title: test_0.py
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/12

import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(10000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()

env.close()


