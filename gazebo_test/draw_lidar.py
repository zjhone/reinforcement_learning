# # Title: draw_liadr.py
# # Author: Zhang Jianghu<zhang_jianghu@163.com>
# # Date: 2022/04/20

import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amt


def open_log(filepath):
    ''' ss '''
    try:
        DATA = list()
        with open(filepath, "r", encoding="utf-8") as f:
            for one_record in f.readlines():
                if one_record == '---\n': continue
                else:
                    one_record= json.loads(one_record)
                    DATA.append(one_record)
                    # print('/scan_ranges(20):', type(one_record), one_record)
            return  DATA

    except Exception as er:
        print("Error!\n {}".format(er))
        exit()

filepath = '/home/zjh/bagfiles/lidar.log'
SCAN_RANGES = open_log(filepath)

speed = 1

def my_func(i):
    if i==int(len(SCAN_RANGES)/speed)-1: exit()
    line.set_ydata(SCAN_RANGES[int(speed*i)])
    return line

def init():
    line.set_ydata(0*(x))
    return line


if __name__=="__main__":

    # a = input('Press any char to play!')

    fig, ax = plt.subplots()
    x = np.arange(0, 20, 1)
    line, = ax.plot(x, np.sin(x))

    anti = amt.FuncAnimation(fig=fig, func=my_func, frames=np.arange(0, int(len(SCAN_RANGES)/speed)),
                             init_func=init, interval=1, blit=False)
    plt.axis([-1,20,-1,8])
    plt.grid()
    plt.show()