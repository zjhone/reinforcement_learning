# Title: tkinter_test.py
# Author: Zhang Jianghu<zhang_jianghu@163.com>
# Date: 2022/03/19

import tkinter

flags = ""


def call(event):
    global flags
    print(event.keysym)
    flags = event.keysym  # 打印按下的键值


win = tkinter.Tk()
frame = tkinter.Frame(win, width=200, height=200)
frame.bind("<Key>", call)  # 触发的函数
print(flags)
frame.focus_set()  # 必须获取焦点
frame.pack()
win.mainloop()

