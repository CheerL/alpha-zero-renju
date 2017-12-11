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
    utils.PAI_SUMMARY_PATH = os.path.join(utils.PAI_ROOT_PATH, 'summary_0.1')
    path_list = [utils.PAI_DB_PATH, utils.PAI_MODEL_PATH, utils.PAI_RECORD_PATH, utils.PAI_SUMMARY_PATH]
    utils.path_init(path_list, True)
    copy_best_model()

def copy_best_model():
    try:
        with tf.gfile.FastGFile(os.path.join(utils.PAI_MODEL_PATH, 'best')) as file:
            best_model_num = int(file.read())
        model_pattern = os.path.join(utils.PAI_MODEL_PATH, 'model-{}*'.format(best_model_num))
        model_path = utils.pai_find_path(model_pattern)
        assert len(model_path) > 0
        utils.pai_model_copy(best_model_num)
    except:
        pass

def main(_):
    pai_constant_init()
    # utils.SAVE_MODEL = True
    net.train(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
