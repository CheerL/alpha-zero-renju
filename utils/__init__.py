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
SUMMARY_PATH = os.path.join(ROOT_PATH, 'summary')

# pai and pai path
USE_PAI = False
PAI_ROOT_PATH = None
PAI_DB_PATH = None
PAI_MODEL_PATH = None
PAI_RECORD_PATH = None
PAI_SUMMARY_PATH = None

# oss_path
OSS_ROOT_PATH = 'alpha/model/'
OSS_MODEL_PATH = OSS_ROOT_PATH + 'model/'
OSS_DB_PATH = OSS_ROOT_PATH + 'db/'
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
SAVE_PSQ = False
SAVE_RECORD = True

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

# train
TRAIN_EPOCH_GAME_NUM = 100
TRAIN_EPOCH_REPEAT_NUM = 1
SUMMARY_INTERVAL = 3
BATCH_SIZE = 100
L2_DECAY = 0.0001
MOMENTUM = 0.9
BASE_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.98
LEARNING_RATE_DECAY_STEP = 1000

# compare
COMPARE_TIME = 20
COMPARE_WIN_RATE = 0.55


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

def pai_model_copy(model_num):
    model_pattern = os.path.join(PAI_MODEL_PATH, 'model-{}*'.format(model_num))
    model_path = pai_find_path(model_pattern)
    assert len(model_path) > 0, 'Model {} does not exist'.format(model_num)
    for model_part in model_path:
        model_part_name = model_part.split('/')[-1]
        model_part_path = os.path.join(MODEL_PATH, model_part_name)
        pai_copy(model_part, model_part_path)

    if not tf.gfile.Exists(os.path.join(MODEL_PATH, 'best')):
        pai_copy(os.path.join(PAI_MODEL_PATH, 'best'), os.path.join(MODEL_PATH, 'best'))
    if not tf.gfile.Exists(os.path.join(MODEL_PATH, 'checkpoint')):
        pai_copy(os.path.join(PAI_MODEL_PATH, 'checkpoint'), os.path.join(MODEL_PATH, 'checkpoint'))

def pai_find_path(pattern):
    return tf.gfile.Glob(pattern)

def pai_read_compare_record(best_num, compare_num):
    compare_record_path = os.path.join(PAI_RECORD_PATH, 'compare-{}-{}'.format(best_num, compare_num))
    try:
        with tf.gfile.GFile(compare_record_path) as file:
            win, total = file.read().split('-')
        return int(win), int(total)
    except:
        with tf.gfile.GFile(compare_record_path, 'w') as file:
            file.write('0-0')
        return 0, 0

def pai_write_compare_record(best_num, compare_num, compare_win):
    compare_record_path = os.path.join(PAI_RECORD_PATH, 'compare-{}-{}'.format(best_num, compare_num))
    win, total = pai_read_compare_record()
    win = win + 1 if compare_win else win
    total = total + 1
    with tf.gfile.GFile(compare_record_path, 'w') as file:
        file.write('{}-{}'.format(win, total))

def pai_change_best(best_num):
    with tf.gfile.FastGFile(os.path.join(PAI_MODEL_PATH, 'best'), 'w') as file:
        file.write(str(best_num))

# init
path_init([LOG_PATH, RECORD_PATH, DB_PATH, MODEL_PATH, SUMMARY_PATH])
