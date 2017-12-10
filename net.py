import os
import numpy as np
import tensorflow as tf
import tflearn as tl
import utils
from utils.logger import Logger
from utils.tfrecord import generate_dataset
from functools import wraps


def no_same_net(NET):
    @wraps(NET)
    def _no_same_net(model_num=-1):
        if model_num not in NET.net_dict:
            new_net = NET(model_num)
            if model_num is -1:
                NET.net_dict[-1] = new_net

            return new_net
        else:
            return NET.net_dict[model_num]
    return _no_same_net

@no_same_net
class Net(object):
    net_dict = {}

    def __init__(self, model_num=-1):
        self.logger = Logger('game')
        self.rot, self.rot_inverse = self.generate_matrix_trans()

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.feature = tf.placeholder(tf.float32, [None, utils.SIZE, utils.SIZE, utils.FEATURE_CHANNEL], name='feature')
            self.expect = tf.placeholder(tf.float32, [None, utils.SIZE ** 2], name='expect')
            self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.train_step = tf.get_variable('train_step', initializer=0, dtype=tf.int32, trainable=False)
            self.epoch = tf.get_variable('epoch', initializer=-1, dtype=tf.int32, trainable=False)
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
            self.load_model(model_num)

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
            weight_decay=utils.L2_DECAY
            )
        ph = tl.layers.normalization.batch_normalization(ph)
        ph = tl.activations.relu(ph)
        ph = tl.layers.core.flatten(ph)
        ph = tl.layers.core.fully_connected(
            ph,
            utils.POLICY_HEAD_FC_DIM_OUT,
            activation='softmax',
            regularizer='L2',
            weight_decay=utils.L2_DECAY
        )
        return ph

    def add_vh(self, net):
        vh = tl.layers.conv.conv_2d(
            net,
            utils.VALUE_HEAD_CONV_DIM_OUT,
            utils.VALUE_HEAD_KERNEL_SIZE,
            regularizer='L2',
            weight_decay=utils.L2_DECAY
            )
        vh = tl.layers.normalization.batch_normalization(vh)
        vh = tl.activations.relu(vh)
        vh = tl.layers.core.flatten(vh)
        vh = tl.layers.core.fully_connected(
            vh,
            utils.VALUE_HEAD_FC_DIM_MID,
            activation='relu',
            regularizer='L2',
            weight_decay=utils.L2_DECAY
            )
        vh = tl.layers.core.fully_connected(
            vh,
            utils.VALUE_HEAD_FC_DIM_OUT,
            activation='tanh',
            regularizer='L2',
            weight_decay=utils.L2_DECAY
            )
        return vh

    def add_accuracy(self, predict, expect):
        # accuracy = tl.metrics.accuracy_op(predict, expect)
        accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predict, tf.argmax(expect, 1), 3), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def add_loss(self, predict, expect, value, reward):
        xent = tl.objectives.categorical_crossentropy(predict, expect)
        square = tf.reduce_mean(tf.square(value - reward))
        l2 = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = utils.XENT_COEF * xent + utils.SQUARE_COEF * square + l2
        tf.summary.scalar('xent', xent)
        tf.summary.scalar('square', square)
        tf.summary.scalar('l2', l2)
        tf.summary.scalar('loss', loss)
        return loss

    def add_trainer(self, loss):
        momentum_optimizer = tl.optimizers.Momentum(
            learning_rate=utils.BASE_LEARNING_RATE,
            momentum=utils.MOMENTUM,
            lr_decay=utils.LEARNING_RATE_DECAY,
            decay_step=utils.LEARNING_RATE_DECAY_STEP
        )
        momentum_optimizer.build(self.train_step)
        return momentum_optimizer.get_tensor().minimize(loss, global_step=self.train_step)

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

    def build(self):
        self.net = self.add_net()

        self.predict = self.add_ph(self.net)
        self.value = self.add_vh(self.net)
        self.accuracy = self.add_accuracy(self.predict, self.expect)
        self.loss = self.add_loss(self.predict, self.expect, self.value, self.reward)
        self.trainer = self.add_trainer(self.loss)
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
            if not self.summary:
                self.summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
            if not self.summary_writer:
                summary_path = utils.PAI_SUMMARY_PATH if utils.USE_PAI else utils.SUMMARY_PATH
                self.summary_writer = tf.summary.FileWriter(summary_path)

            iterator, next_batch = generate_dataset(files, batch_size)
            for epoch in range(utils.TRAIN_EPOCH_REPEAT_NUM):
                self.sess.run(iterator.initializer)
                self.logger.info('Start train epoch {}/{}'.format(
                    epoch + 1, utils.TRAIN_EPOCH_REPEAT_NUM))
                while True:
                    try:
                        feature, expect, reward = self.sess.run(next_batch)
                        _, train_step = self.sess.run(
                            [self.trainer, self.train_step],
                            feed_dict={
                                self.feature: feature,
                                self.expect: expect,
                                self.reward: reward
                            })

                        if train_step % utils.SUMMARY_INTERVAL == 0:
                            p, v, summary = self.sess.run(
                                [self.predict, self.value, self.summary],
                                feed_dict={
                                    self.feature: feature,
                                    self.expect: expect,
                                    self.reward: reward
                                })
                            self.summary_writer.add_summary(summary, train_step)
                            self.logger.info('Save summary {}'.format(train_step))
                            print(p[3], v[3])
                    except tf.errors.OutOfRangeError:
                        break
            self.logger.info('Training end')

    def model_path(self, suffix, pai=True):
        if utils.USE_PAI and pai:
            return os.path.join(utils.PAI_MODEL_PATH, suffix)
        else:
            return os.path.join(utils.MODEL_PATH, suffix)

    def exist_model(self, model_num):
        return tf.gfile.Exists(self.model_path('model-{}.index'.format(model_num)))

    def load_model(self, model_num):
        assert isinstance(model_num, int), 'model num must be int'

        if model_num is -1:
            self.logger.info('Try to load best model')
            try:
                with tf.gfile.FastGFile(self.model_path('best')) as file:
                    model_num = int(file.read())
                    assert self.exist_model(model_num), 'Best model {} does not exist'.format(model_num)
                    self.logger.info('Best model is {}'.format(model_num))
            except Exception as e:
                self.logger.error(e)
                model_num = 0

            self.load_model(model_num)

        else:
            if self.exist_model(model_num):
                self.saver.restore(self.sess, self.model_path('model-{}'.format(model_num)))
                self.logger.info('Load model {}'.format(model_num))
                self.net_dict[model_num] = self
            elif model_num is 0:
                self.sess.run(tf.global_variables_initializer())
                self.logger.info('Build init model 0')
                self.save_model(True)
            else:
                self.logger.info('Model {} no exist, try to load best model'.format(model_num))
                self.load_model(-1)

    def save_model(self, write_best_record=False):
        model_num = self.sess.run(self.epoch.assign_add(1))
        self.saver.save(self.sess, self.model_path('model', True), self.epoch)
        self.net_dict[model_num] = self
        self.logger.info('Save model {}'.format(model_num))

        if model_num > 0:
            del self.net_dict[model_num - 1]

        if write_best_record:
            with tf.gfile.FastGFile(self.model_path('best', True), 'w') as file:
                file.write(str(model_num))
            self.logger.info('Best model is {}'.format(model_num))
            self.net_dict[-1] = self

    def get_model_num(self):
        return self.sess.run(self.epoch)


def main():
    model_num = 0
    db_path = utils.PAI_DB_PATH if utils.USE_PAI else utils.DB_PATH
    records = utils.pai_find_path(os.path.join(db_path, 'game-{}*'.format(model_num)))
    net = Net(model_num)
    net.train(records)
    # net.save_model()

if __name__ == '__main__':
    main()
