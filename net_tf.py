import numpy as np
import tensorflow as tf
import utils

class Net(object):
    def __init__(self):
        self.feature = tf.placeholder(tf.float32, [None, utils.SIZE, utils.SIZE, utils.FEATURE_CHANNEL])
        self.expect = tf.placeholder(tf.float32, [None, utils.SIZE, utils.SIZE])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.predict = None
        self.value = None
        self.loss = None
        self.accuracy = None
        self.is_train = True
        self.conv_num = 0
        self.res_num = 0
        self.sess = tf.Session()

    def weight_generater(self, shape, dtype=tf.float32):
        return tf.Variable(tf.truncated_normal(shape, dtype=dtype))

    def bais_generator(self, shape, dtype=tf.float32):
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype))

    def batch_norm(self, data, eps=1e-05, decay=0.9, name=None):
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = data.shape[-1:]
            axis = list(range(len(data.shape) - 1))
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                mean, var = tf.nn.moments(data, axis, name='moments')
                ema_apply_op = ema.apply([mean, var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(mean), tf.identity(var)

            def mean_var_without_update():
                mean = tf.get_variable('mean', params_shape,
                                       initializer=tf.zeros_initializer,
                                       trainable=False)
                var = tf.get_variable('variance', params_shape,
                                      initializer=tf.ones_initializer,
                                      trainable=False)
                return mean, var

            mean, var = tf.cond(tf.constant(self.is_train), mean_var_with_update, mean_var_without_update)

            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
            return tf.nn.batch_normalization(data, mean, var, beta, gamma, eps)

    def fc_block(self, data, dim_out, name=None):
        with tf.variable_scope(name, default_name='Fc'):
            if len(data.shape) is 4:
                fc_flat_dim = data.shape[1].value * data.shape[2].value * data.shape[3].value
            else:
                fc_flat_dim = data.shape[1].value

            fc_data_flat = tf.reshape(data, [-1, fc_flat_dim])
            fc_weight = self.weight_generater([fc_flat_dim, dim_out])
            return tf.matmul(fc_data_flat, fc_weight)

    def conv_block(self, data, dim_out, kernel_size, with_relu=True,
                   strides=[1, 1, 1, 1], padding="SAME", name=None):
        name = 'Conv{}'.format(self.conv_num) if not name else name
        with tf.variable_scope(name, default_name='Conv'):
            dim_in = data.shape[-1].value
            kernel = self.weight_generater([kernel_size, kernel_size, dim_in, dim_out])
            conv = tf.nn.conv2d(data, kernel, strides, padding, name=name)
            bn = self.batch_norm(conv, self.is_train)
            conv_out = tf.cond(tf.constant(with_relu), lambda: tf.nn.relu(bn), lambda: bn)
            self.conv_num += 1
            return conv_out

    def res_block(self, data, dim_mid, dim_out, kernel_size, name=None):
        name = 'Res{}'.format(self.res_num) if not name else name
        with tf.variable_scope(name, default_name='Res'):
            conv_0 = self.conv_block(data, dim_mid, kernel_size)
            conv_1 = self.conv_block(conv_0, dim_out, kernel_size, with_relu=False)
            res_highway = tf.add(data, conv_1)
            res_out = tf.nn.relu(res_highway)
            self.res_num += 1
            return res_out

    def policy_head(self, data, dim_out, fc_dim_out, kernel_size):
        with tf.variable_scope('Policy_head'):
            ph_conv = self.conv_block(data, dim_out, kernel_size)
            ph_fc = self.fc_block(ph_conv, fc_dim_out)
            predict = tf.nn.softmax(ph_fc)
            return predict

    def value_head(self, data, dim_out, fc_dim_mid, fc_dim_out, kernel_size):
        with tf.variable_scope('Value_head'):
            vh_conv = self.conv_block(data, dim_out, kernel_size)
            vh_fc_0 = self.fc_block(vh_conv, fc_dim_mid)
            vh_relu = tf.nn.relu(vh_fc_0)
            vh_fc_1 = self.fc_block(vh_relu, fc_dim_out)
            value = tf.nn.tanh(vh_fc_1)
            return value

    def accuracy_block(self, predict, expect):
        with tf.variable_scope('Accuracy'):
            correct = tf.equal(tf.argmax(predict, 1), tf.argmax(expect, 1))
            return tf.reduce_mean(tf.cast(correct, tf.float32))

    def loss_block(self, predict, expect, value, reward):
        with tf.variable_scope('Loss'):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=expect, logits=predict
            ))
            dis = tf.sqrt(tf.reduce_sum(tf.square(value - reward)))
            loss = tf.add(xent, dis)
            return loss

    def build(self):
        res_in = self.conv_block(self.feature, utils.FILTER_NUM, utils.CONV_KERNEL_SIZE)

        for _ in range(utils.RES_BLOCK_NUM):
            res_out = self.res_block(
                res_in,
                utils.FILTER_NUM,
                utils.FILTER_NUM,
                utils.CONV_KERNEL_SIZE
                )
            res_in = res_out

        self.predict = self.policy_head(
            res_out,
            utils.POLICY_HEAD_CONV_DIM_OUT,
            utils.POLICY_HEAD_FC_DIM_OUT,
            utils.POLICY_HEAD_KERNEL_SIZE
            )
        self.value = self.value_head(
            res_out,
            utils.VALUE_HEAD_CONV_DIM_OUT,
            utils.VALUE_HEAD_FC_DIM_MID,
            utils.VALUE_HEAD_FC_DIM_OUT,
            utils.VALUE_HEAD_KERNEL_SIZE
            )
        self.accuracy = self.accuracy_block(self.predict, self.expect)
        self.loss = self.loss_block(self.predict, self.expect, self.value, self.reward)

        self.sess.run(tf.global_variables_initializer())

    def get_predict_and_value(self, feature):
        if len(feature.shape) is not 4:
            feature = tf.reshape(feature, [-1, utils.SIZE, utils.SIZE, utils.FEATURE_CHANNEL])

        return self.sess.run([self.predict, self.value], feed_dict={self.feature: feature})

def main():
    import time
    times = 20
    net = Net()
    net.build()
    f = np.random.sample((1, utils.SIZE, utils.SIZE, utils.FEATURE_CHANNEL))
    st = time.time()
    for _ in range(times):
        net.get_predict_and_value(f)
    et = time.time()
    print("{}".format((et - st) / times))


if __name__ == '__main__':
    main()