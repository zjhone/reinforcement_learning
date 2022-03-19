
import sys
import pygame
from pygame.locals import *

#初始化

pygame.init()

##变量存放处

size = width, height = 150,150
bgColor = (0,0,0)

##設置界面寬高

screen = pygame.display.set_mode(size)

##設置標題
pygame.display.set_caption("Pygame事件")

##要在Pygame中使用文本，必须创建Font对象

##第一个参数指定字体 ，第二个参数指定字体大小

font =pygame.font.Font(None,40)

##调用get_linesize()方法获得每行文本的高度

line_height = font.get_linesize()
position = 0
screen.fill(bgColor)

while True:
      for event in pygame.event.get():
            if event.type == pygame.QUIT:
                  sys.exit()

            flags = ""
            if event.type == KEYDOWN:
                  if event.key == K_LEFT:
                        flags = "left"

                  elif event.key == K_RIGHT:
                        flags = "right"

                  elif event.key == K_UP:
                        flags = "up"

                  elif event.key == K_DOWN:
                        flags ="down"

                  # render()将文本渲染成Surface对象, 第一个参数是带渲染的文本,
                  # 第二个参数指定是否消除锯齿, 第三个参数指定文本的颜色
                  screen.blit(font.render(flags, True, (254, 0, 0)) , (0, position))
                  # print(str(event))
                  position += line_height
                  if position >= height:
                        position = 0
                        screen.fill(bgColor)

      pygame.display.flip()
