# -*- coding:utf-8 -*-
import sys
import os
from collections import namedtuple

# path
if getattr(sys, 'frozen', False):
    ROOT_PATH = os.path.dirname(os.path.dirname(sys.executable))
else:
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_PATH = os.path.join(ROOT_PATH, 'log')
RECORD_PATH = os.path.join(ROOT_PATH, 'record')
DB_PATH = os.path.join(ROOT_PATH, 'db')

for path in [LOG_PATH, RECORD_PATH, DB_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)

# move structure
Move = namedtuple('Move', ['x', 'y'])

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

# network
BOARD_HISTORY_LENGTH = 5
FEATURE_CHANNEL = 2 * BOARD_HISTORY_LENGTH + 1
CONV_KERNEL_SIZE = 3
FILTER_NUM = 256
RES_BLOCK_NUM = 19
POLICY_HEAD_CONV_DIM_OUT = 2
POLICY_HEAD_KERNEL_SIZE = 1
POLICY_HEAD_FC_DIM_OUT = SIZE ** 2
VALUE_HEAD_CONV_DIM_OUT = 1
VALUE_HEAD_KERNEL_SIZE = 1
VALUE_HEAD_FC_DIM_MID = FILTER_NUM
VALUE_HEAD_FC_DIM_OUT = 1
