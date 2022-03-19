# Title: getchar.py
# Author: Zhang Jianghu
# Date: 2022/03/19

import sys
from Cython.Plex.Regexps import EOF

def getchar(input_str=""):
    print(input_str)
    return sys.stdin.read(1) # reads one byte at a time, similar to getchar()

