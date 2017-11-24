# -*- coding:utf-8 -*-
import sys
import os

# path
if getattr(sys, 'frozen', False):
    ROOT_PATH = os.path.dirname(os.path.dirname(sys.executable))
else:
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_PATH = os.path.join(ROOT_PATH, 'log')
RECORD_PATH = os.path.join(ROOT_PATH, 'record')

for path in [LOG_PATH, RECORD_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)

# color
BLACK = 1
WHITE = -1
EMPTY = 0
COLOR = {
    BLACK: 'Black',
    WHITE: 'White'
}

# Player type
HUMAN, GOMOCUP, MCTS, RANDOM = 0, 1, 2, 3

# game default para
SIZE = 20
WIN_NUM = 5

# socket
HOST = 'localhost'
PORT = 10329

