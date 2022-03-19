# Title: policy_iterate.py  策略迭代算法
# Author: Zhang Jianghu
# Date: 2022/03/12

import numpy as np
import gym
import time
import sys
import pygame
from pygame.locals import *


pygame.init()
size = width, height = 150,150
bgColor = (255,255,255)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("键盘监控")
font =pygame.font.Font(None, 30)
line_height = font.get_linesize()
position = 0
screen.fill(bgColor)

if __name__ == "__main__":
    env = gym.make("GridWorld-v0")
    env.reset()
    env.render()
    print(env.gett())
    print("Please input direction:")
    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                sys.exit()

            flags = ""
            if event.type == KEYDOWN:
                if event.key == K_LEFT:
                    flags = "w"
                elif event.key == K_RIGHT:
                    flags = "e"
                elif event.key == K_UP:
                    flags = "n"
                elif event.key == K_DOWN:
                    flags = "s"

                screen.blit(font.render(flags, True, (0, 0, 0)), (0, position))
                position += line_height
                if position >= height:
                    position = 0
                    screen.fill(bgColor)
                next_state, r, is_terminal, temp = env.step(flags)
                print(next_state, r, is_terminal, temp)
                env.render()
                if is_terminal:
                    exit()
        pygame.display.flip()
