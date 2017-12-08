# -*- coding:utf-8 -*-
import sys
import os
import gc
import tensorflow as tf
from collections import namedtuple

# move structure
Move = namedtuple('Move', ['x', 'y'])

# path
if getattr(sys, 'frozen', False):
    ROOT_PATH = os.path.dirname(os.path.dirname(sys.executable))
else:
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_PATH = os.path.join(ROOT_PATH, 'log')
RECORD_PATH = os.path.join(ROOT_PATH, 'record')
DB_PATH = os.path.join(ROOT_PATH, 'db')
MODEL_PATH = os.path.join(ROOT_PATH, 'model')

# pai and pai path
USE_PAI = False
PAI_ROOT_PATH = None
PAI_DB_PATH = None
PAI_MODEL_PATH = None
PAI_RECORD_PATH = None

# oss_path
OSS_ROOT_PATH = 'alpha/model/'
OSS_MODEL_PATH = OSS_ROOT_PATH + 'model/'
OSS_DB_PATH  = OSS_ROOT_PATH + 'db/'
OSS_RECORD_PATH = OSS_ROOT_PATH + 'record/'

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
FULL_SIZE = SIZE ** 2
WIN_NUM = 5

# socket
HOST = 'localhost'
PORT = 10329

# MCTS
MAX_MCTS_EVALUATE_TIME = 100
TAU_CHANGE_ROUND = 0
TAU_UP = 1.0
TAU_LOW = 0.01

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

TRAIN_EPOCH_GAME_NUM = 500

# function
CLEAR = gc.collect

def path_init(paths, pai_path=False):
    for path in paths:
        if pai_path:
            if not tf.gfile.Exists(path):
                tf.gfile.MakeDirs(path)
        else:
            if not os.path.exists(path):
                os.makedirs(path)

def pai_copy(from_path, to_path, overwrite=True):
    tf.gfile.Copy(from_path, to_path, overwrite=overwrite)

def pai_dir_copy(from_path, to_path, pattern='*', overwrite=True):
    for file in pai_find_path(os.path.join(from_path, pattern)):
        file_name = file.split('/')[-1]
        file_path = os.path.join(to_path, file_name)
        pai_copy(file, file_path, overwrite)

def pai_find_path(pattern):
    return tf.gfile.Glob(pattern)

# init
path_init([LOG_PATH, RECORD_PATH, DB_PATH, MODEL_PATH])
