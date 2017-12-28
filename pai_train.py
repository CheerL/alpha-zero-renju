from __future__ import division, print_function, absolute_import

import os
import sys
import argparse
import tensorflow as tf
import utils
import net

FLAGS = None

def pai_constant_init():
    utils.USE_PAI = True
    utils.PAI_ROOT_PATH = FLAGS.checkpointDir
    utils.PAI_DB_PATH = os.path.join(utils.PAI_ROOT_PATH, 'db')
    utils.PAI_MODEL_PATH = os.path.join(utils.PAI_ROOT_PATH, 'model')
    utils.PAI_RECORD_PATH = os.path.join(utils.PAI_ROOT_PATH, 'record')
    utils.PAI_SUMMARY_PATH = os.path.join(utils.PAI_ROOT_PATH, 'summary_all')
    path_list = [utils.PAI_DB_PATH, utils.PAI_MODEL_PATH, utils.PAI_RECORD_PATH, utils.PAI_SUMMARY_PATH]
    utils.path_init(path_list, True)

def main(_):
    pai_constant_init()
    utils.SAVE_MODEL = False
    while True:
        model_num = 0
        net.verificate(model_num)
        net.train(model_num)
        if utils.SAVE_MODEL:
            net.Net.net_dict[model_num] = net.Net.net_dict[model_num + 1]
            net.verificate(model_num)
            del net.Net.net_dict[model_num]
        else:
            net.verificate(model_num)

        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
