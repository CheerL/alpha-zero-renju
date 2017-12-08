import os
import numpy as np
import tensorflow as tf
import tflearn as tl
import utils
from utils.logger import Logger
from utils.tfrecord import generate_dataset
from functools import wraps


def no_same_net(net):
    net_dict = {}
    @wraps(net)
    def _no_same_net(model_num=-1):
        if model_num not in net_dict:
            new_net = net(model_num)
            new_net.net_dict = net_dict

            if model_num is -1:
                net_dict[-1] = new_net
            net_dict[new_net.model_num] = new_net
            return new_net
        else:
            return net_dict[model_num]
    return _no_same_net

@no_same_net
class Net(object):
    def __init__(self, model_num=-1):
        self.logger = Logger('game')
        self.model_num = model_num
        self.rot, self.rot_inverse = self.generate_matrix_trans()
        self.net_dict = None

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.feature = tf.placeholder(tf.float32, [None, utils.SIZE, utils.SIZE, utils.FEATURE_CHANNEL])
            self.expect = tf.placeholder(tf.float32, [None, utils.SIZE ** 2])
            self.reward = tf.placeholder(tf.float32, [None, 1])
            self.predict = None
            self.value = None
            self.net = None
            self.loss = None
            self.accuracy = None
            self.trainer = None
            self.summary = None
            self.summary_writer = None
            self.build()

            self.saver = tf.train.Saver()
            self.load_model()

    def generate_matrix_trans(self):
        rotat_0 = lambda m, axes=(0, 1): m
        rotat_90 = lambda m, axes=(0, 1): np.rot90(m, 1, axes=axes)
        rotat_180 = lambda m, axes=(0, 1): np.rot90(m, 2, axes=axes)
        rotat_270 = lambda m, axes=(0, 1): np.rot90(m, 3, axes=axes)
        reflect_0 = lambda m, axes=(0, 1): np.flip(m, axis=axes[1])
        reflect_90 = lambda m, axes=(0, 1): np.flip(rotat_90(m, axes=axes), axis=axes[1])
        reflect_180 = lambda m, axes=(0, 1): np.flip(rotat_180(m, axes=axes), axis=axes[1])
        reflect_270 = lambda m, axes=(0, 1): np.flip(rotat_270(m, axes=axes), axis=axes[1])

        rot = {
            0: rotat_0,
            1: rotat_90,
            2: rotat_180,
            3: rotat_270,
            4: reflect_0,
            5: reflect_90,
            6: reflect_180,
            7: reflect_270
        }
        rot_inverse = {
            0: rotat_0,
            1: rotat_270,
            2: rotat_180,
            3: rotat_90,
            4: reflect_0,
            5: reflect_90,
            6: reflect_180,
            7: reflect_270
        }
        return rot, rot_inverse

    def add_ph(self, net):
        ph = tl.layers.conv.conv_2d(
            net,
            utils.POLICY_HEAD_CONV_DIM_OUT,
            utils.POLICY_HEAD_KERNEL_SIZE,
            regularizer='L2',
            weight_decay=0.0001
            )
        ph = tl.layers.normalization.batch_normalization(ph)
        ph = tl.activations.relu(ph)
        ph = tl.layers.core.flatten(ph)
        ph = tl.layers.core.fully_connected(
            ph,
            utils.POLICY_HEAD_FC_DIM_OUT,
            activation='softmax',
            regularizer='L2',
            weight_decay=0.0001
        )
        return ph

    def add_vh(self, net):
        vh = tl.layers.conv.conv_2d(
            net,
            utils.VALUE_HEAD_CONV_DIM_OUT,
            utils.VALUE_HEAD_KERNEL_SIZE,
            regularizer='L2',
            weight_decay=0.0001
            )
        vh = tl.layers.normalization.batch_normalization(vh)
        vh = tl.activations.relu(vh)
        vh = tl.layers.core.flatten(vh)
        vh = tl.layers.core.fully_connected(
            vh,
            utils.VALUE_HEAD_FC_DIM_MID,
            activation='relu',
            regularizer='L2',
            weight_decay=0.0001
            )
        vh = tl.layers.core.fully_connected(
            vh,
            utils.VALUE_HEAD_FC_DIM_OUT,
            activation='tanh',
            regularizer='L2',
            weight_decay=0.0001
            )
        return vh

    def add_accuracy(self, predict, expect):
        return tl.metrics.accuracy_op(predict, expect)

    def add_loss(self, predict, expect, value, reward):
        xent = tl.objectives.categorical_crossentropy(predict, expect)
        square = tf.reduce_sum(tf.square(value - reward))
        l2 = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # l2 = tf.reduce_sum(tf.square(
        # tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) / utils.L2_DECAY)) * utils.L2_DECAY
        return xent + square + l2

    def add_trainer(self, loss):
        momentum_optimizer = tl.optimizers.Momentum(
            learning_rate=utils.BASE_LEARNING_RATE,
            momentum=utils.MOMENTUM,
            lr_decay=utils.LEARNING_RATE_DECAY,
            decay_step=utils.LEARNING_RATE_DECAY_STEP
        )
        momentum_optimizer.build(True)
        return momentum_optimizer.get_tensor().minimize(loss)

    def add_net(self):
        net = tl.layers.core.input_data(placeholder=self.feature)
        net = tl.layers.conv.conv_2d(
            net,
            utils.FILTER_NUM,
            utils.CONV_KERNEL_SIZE,
            regularizer='L2',
            weight_decay=utils.L2_DECAY
            )
        net = tl.layers.normalization.batch_normalization(net)
        net = tl.activations.relu(net)

        for _ in range(utils.RES_BLOCK_NUM):
            net = tl.layers.conv.residual_block(net, 1, utils.FILTER_NUM)
            net = tl.activations.relu(net)

        return net

    def add_summary(self, loss, accuracy):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        return tf.summary.FileWriter(utils.SUMMARY_PATH, self.graph), tf.summary.merge_all()

    def build(self):
        self.net = self.add_net()

        self.predict = self.add_ph(self.net)
        self.value = self.add_vh(self.net)
        self.accuracy = self.add_accuracy(self.predict, self.expect)
        self.loss = self.add_loss(self.predict, self.expect, self.value, self.reward)
        self.trainer = self.add_trainer(self.loss)
        self.summary_writer, self.summary = self.add_summary(self.loss, self.accuracy)

        self.logger.info('Build net successfully')

    def get_predict_and_value(self, feature, rot=False):
        if not rot:
            predict, value = self.sess.run(
                [self.predict, self.value],
                feed_dict={self.feature: feature.astype(np.float32)}
                )
            predict = predict[0]
            value = value[0, 0]
            return predict, value
        else:
            rot_num = np.random.randint(0, 8)
            feature = self.rot[rot_num](feature, (1, 2))
            predict, value = self.sess.run(
                [self.predict, self.value],
                feed_dict={self.feature: feature.astype(np.float32)}
                )
            predict = self.rot_inverse[rot_num](
                np.reshape(predict, (utils.SIZE, utils.SIZE))
            ).reshape(utils.FULL_SIZE)
            value = value[0, 0]
            return predict, value

    def train(self, files, batch_size=utils.BATCH_SIZE):
        with self.graph.as_default():
            iterator, next_batch = generate_dataset(files, batch_size)
            train_time = 0
            for epoch in range(utils.TRAIN_EPOCH_REPEAT_NUM):
                self.sess.run(iterator.initializer)
                self.logger.info('Start train epoch {}/{}'.format(
                    epoch + 1, utils.TRAIN_EPOCH_REPEAT_NUM))
                while True:
                    try:
                        for _ in range(utils.SUMMARY_INTERVAL):
                            feature, expect, reward = self.sess.run(next_batch)
                            self.sess.run(
                                self.trainer,
                                feed_dict={
                                    self.feature: feature,
                                    self.expect: expect,
                                    self.reward: reward
                                })
                            train_time += 1
                        else:
                            feature, expect, reward = self.sess.run(next_batch)
                            summary = self.sess.run(
                                self.summary,
                                feed_dict={
                                    self.feature: feature,
                                    self.expect: expect,
                                    self.reward: reward
                                })
                            self.summary_writer.add_summary(summary, train_time)
                            self.logger.info('Save summary')
                    except tf.errors.OutOfRangeError:
                        break
            self.logger.info('Training end')

    def model_path(self, suffix, pai=False):
        if utils.USE_PAI:
            if pai:
                return os.path.join(utils.PAI_MODEL_PATH, suffix)
            else:
                return os.path.join(utils.MODEL_PATH, suffix)
        else:
            return os.path.join(utils.MODEL_PATH, suffix)

    def exsit_model(self):
        if self.model_num is -1:
            return tf.gfile.Exists(self.model_path('model-best.index'))
        else:
            return tf.gfile.Exists(self.model_path('model-{}.index'.format(self.model_num)))

    def load_model(self):
        assert isinstance(self.model_num, int), 'model num must be int'

        if self.model_num is -1:
            self.logger.info('Try to load best model')
            if tf.gfile.Exists(self.model_path('best')):
                try:
                    with tf.gfile.FastGFile(self.model_path('best')) as file:
                        self.model_num = int(file.read())
                        self.logger.info('Best model is {}'.format(self.model_num))
                        assert self.exsit_model(), 'Best model {} does not exist'.format(self.model_num)
                except Exception as e:
                    self.logger.error(e)
                    self.model_num = 0

            else:
                self.logger.info('Best record does not exsit')
                self.model_num = 0

            self.load_model()

        else:
            if self.exsit_model():
                self.saver.restore(self.sess, self.model_path('model-{}'.format(self.model_num)))
                self.logger.info('Load model {}'.format(self.model_num))
            elif self.model_num is 0:
                self.sess.run(tf.global_variables_initializer())
                self.logger.info('Build init model 0')
                self.save_model(True)
            else:
                self.logger.info('Model {} no exist, try to load best model'.format(self.model_num))
                self.model_num = -1
                self.load_model()

    def save_model(self, write_best_record=False):
        self.saver.save(self.sess, self.model_path('model'), self.model_num)
        self.logger.info('Save model {}'.format(self.model_num))

        if write_best_record:
            with tf.gfile.FastGFile(self.model_path('best'), 'w') as file:
                file.write(str(self.model_num))
            self.logger.info('Best model is {}'.format(self.model_num))

        if utils.USE_PAI:
            pattern = 'model-{}*'.format(self.model_num)
            utils.pai_dir_copy(utils.MODEL_PATH, utils.PAI_MODEL_PATH, pattern)
            utils.pai_copy(self.model_path('checkpoint'), self.model_path('checkpoint', True))
            if write_best_record:
                with tf.gfile.FastGFile(self.model_path('best', True), 'w') as file:
                    file.write(str(self.model_num))

    def change_model_num(self, new_model_num):
        assert isinstance(new_model_num, int), 'Model num must be int'
        del self.net_dict[self.model_num]
        self.net_dict[new_model_num] = self
        self.model_num = new_model_num
        self.logger.info('Change model num from {} to {}'.format(self.model_num, new_model_num))
        self.save_model()


def main():
    db_path = utils.PAI_DB_PATH if utils.USE_PAI else utils.DB_PATH
    records = utils.pai_find_path(os.path.join(db_path, '*'))
    net = Net()
    net.train(records)
    net.change_model_num(net.model_num + 1)

if __name__ == '__main__':
    main()
