from __future__ import division, print_function, absolute_import

import os
import sys
import argparse
import tensorflow as tf
import net
import utils
from utils.bayesian_opt import BayesianOptimization

FLAGS = None

def pai_constant_init():
    utils.USE_PAI = True
    utils.PAI_ROOT_PATH = FLAGS.checkpointDir
    utils.PAI_DB_PATH = os.path.join(utils.PAI_ROOT_PATH, 'db')
    utils.PAI_MODEL_PATH = os.path.join(utils.PAI_ROOT_PATH, 'model')
    utils.PAI_RECORD_PATH = os.path.join(utils.PAI_ROOT_PATH, 'record')
    utils.PAI_SUMMARY_PATH = os.path.join(utils.PAI_ROOT_PATH, 'summary')
    path_list = [utils.PAI_DB_PATH, utils.PAI_MODEL_PATH, utils.PAI_RECORD_PATH, utils.PAI_SUMMARY_PATH]
    utils.path_init(path_list, True)

def bayes_func(repeat_time, learning_rate, batch_size):
    utils.TRAIN_EPOCH_REPEAT_NUM = int(repeat_time)
    utils.BASE_LEARNING_RATE = learning_rate
    utils.BATCH_SIZE = int(batch_size)
    utils.SAVE_MODEL = False
    model_num = 6
    _net = net.Net(model_num)
    _net.load_model(model_num)
    net.train(model_num, write_summary=False)
    acc, loss = net.verificate(model_num)
    return -loss

def main(_):
    pai_constant_init()
    params = {
        'repeat_time': (0.5, 10.5),
        'learning_rate': (1e-7, 0.1),
        'batch_size': (10, 150)
    }
    explore_params = {
        'repeat_time': [0],
        'learning_rate': [1e-5],
        'batch_size': [64],
    }
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    bo = BayesianOptimization(bayes_func, params)
    bo.explore(explore_params)
    bo.maximize(init_points=5, n_iter=25, acq='ei', xi=0.05, **gp_params)
    print(bo.res['max'])
    print(bo.res['all'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
