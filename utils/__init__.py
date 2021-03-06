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
HUMAN, GOMOCUP, MCTS, RANDOM, TRANS = 0, 1, 2, 3, 4

# game default para
SIZE = 20
FULL_SIZE = SIZE ** 2
WIN_NUM = 5
SAVE_PSQ = False
SAVE_RECORD = True
SAVE_MODEL = False

# socket
HOST = 'localhost'
PORT = 10329

# MCTS
C_PUCT = 3
MAX_MCTS_EVALUATE_TIME = 1
TAU_CHANGE_ROUND = 30
TAU_UP = 1.0
TAU_LOW = 0.05
NOISE_RATE = 0.25

# network
BOARD_HISTORY_LENGTH = 5
FEATURE_CHANNEL = 2 * BOARD_HISTORY_LENGTH
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
VERIFICATION_GAME_NUM = 2
TRAIN_EPOCH_GAME_NUM = 20
TRAIN_EPOCH_REPEAT_NUM = 500
TRAIN_SAMPLE_NUM = 2 * 10
SUMMARY_INTERVAL = 5
BATCH_SIZE = 150
L2_DECAY = 1e-4
MOMENTUM = 0.9
BASE_LEARNING_RATE = 2e-5
XENT_COEF = 1
SQUARE_COEF = 0.1

# compare
COMPARE_TIME = 5
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

def pai_open(path, tag):
    if USE_PAI:
        return tf.gfile.FastGFile(path, tag)
    else:
        return open(path, tag)

def pai_copy(from_path, to_path, overwrite=True):
    tf.gfile.Copy(from_path, to_path, overwrite=overwrite)

def pai_dir_copy(from_path, to_path, pattern='*', overwrite=True):
    for file in pai_find_path(os.path.join(from_path, pattern)):
        file_name = file.split('/')[-1]
        file_path = os.path.join(to_path, file_name)
        pai_copy(file, file_path, overwrite)

def pai_model_copy(model_num):
    local_model_pattern = os.path.join(MODEL_PATH, 'model-{}*'.format(model_num))
    if not pai_find_path(local_model_pattern):
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
    compare_record_path = os.path.join(PAI_RECORD_PATH, 'compare-{}-{}'.format(compare_num, best_num))
    try:
        with tf.gfile.GFile(compare_record_path) as file:
            win, total = file.read().split('-')
        return int(win), int(total)
    except:
        with tf.gfile.GFile(compare_record_path, 'w') as file:
            file.write('0-0')
        return 0, 0

def pai_write_compare_record(best_num, compare_num, compare_win):
    compare_record_path = os.path.join(PAI_RECORD_PATH, 'compare-{}-{}'.format(compare_num, best_num))
    win, total = pai_read_compare_record(best_num, compare_num)
    win = win + 1 if compare_win else win
    total = total + 1
    with tf.gfile.GFile(compare_record_path, 'w') as file:
        file.write('{}-{}'.format(win, total))

def pai_change_best(best_num, prefix=''):
    model_path = PAI_MODEL_PATH if USE_PAI else MODEL_PATH
    file_name = 'best' if prefix == '' else '-'.join([prefix, 'best'])
    with tf.gfile.FastGFile(os.path.join(model_path, file_name), 'w') as file:
        file.write(str(best_num))

def pai_read_best(prefix=''):
    model_path = PAI_MODEL_PATH if USE_PAI else MODEL_PATH
    file_name = 'best' if prefix == '' else '-'.join([prefix, 'best'])
    with tf.gfile.FastGFile(os.path.join(model_path, file_name), 'r') as file:
        return int(file.read())

def pai_win_rate_record(model_num, color):
    with tf.gfile.FastGFile(os.path.join(PAI_RECORD_PATH, 'winrate-{}'.format(model_num)), 'w+') as file:
        try:
            black_win, white_win = map(int, file.read().split('-'))
        except:
            black_win, white_win = 0, 0

        if color is BLACK:
            black_win += 1
        elif color is WHITE:
            white_win += 1
        file.write('{}-{}'.format(black_win, white_win))

# init
path_init([LOG_PATH, RECORD_PATH, DB_PATH, MODEL_PATH, SUMMARY_PATH])
