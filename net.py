import os
import utils
import numpy as np
import time

from functools import wraps
from caffe2.python import core, workspace, model_helper, brew, utils as caffe2_utils
from caffe2.proto import caffe2_pb2

def write_db(db_type, db_name, feature, expect, reward):
    db_path = os.path.join(utils.DB_PATH, db_name)
    db = core.C.create_db(db_type, db_path, core.C.Mode.write)
    trans = db.new_transaction()
    label = np.argmax(expect, axis=1)

    key = str(time.time()).replace('.', '')[2:]
    for i in range(feature.shape[0]):
        proto = caffe2_pb2.TensorProtos()
        proto.protos.extend([
            caffe2_utils.NumpyArrayToCaffe2Tensor(feature[i]),
            caffe2_utils.NumpyArrayToCaffe2Tensor(expect[i]),
            caffe2_utils.NumpyArrayToCaffe2Tensor(reward[i]),
            caffe2_utils.NumpyArrayToCaffe2Tensor(label[i])
        ])
        trans.put(
            '{}{}'.format(key, i),
            proto.SerializeToString()
        )

    trans.commit()
    db.close()
    del db
    del trans

def check_workspace(func):
    @wraps(func)
    def after_check_func(self, *args, **kw):
        if workspace.CurrentWorkspace() != self.WORKSPACE:
            workspace.SwitchWorkspace(self.WORKSPACE, True)
        return func(self, *args, **kw)
    return after_check_func

class Net(object):
    WORKSPACE = 'default'
    ARG = {"order": "NCHW"}

    def __init__(self, name, batch_size, board_size):
        self.batch_size = batch_size
        self.model = model_helper.ModelHelper(name=name, arg_scope=self.ARG)
        self.conv_count = 0
        self.res_count = 0
        self.board_size = board_size
        self.board_full_size = board_size ** 2
        self.filter_num = 256
        self.res_block_num = 19
        self.feature_channels = utils.BOARD_HISTORY_LENGTH * 2 + 1
        self.base_lr = -0.1
        self.lr_policy = 'step'
        self.stepsize = 2000000
        self.gamma = 0.1

    @check_workspace
    def add_db_input(self, db_name, db_type):
        db_path = os.path.join(utils.DB_PATH, '{}.db'.format(db_name))
        _feature, _expect, _reward, _label = self.model.TensorProtosDBInput(
            [db_reader]
            ['_feature', '_expect', '_reward', '_label'],
            batch_size=self.batch_size,
            db=db_path,
            db_type=db_type
            )

        feature = self.model.Cast(_feature, 'feature', to=core.DataType.FLOAT)
        expect = self.model.Cast(_expect, 'expect', to=core.DataType.FLOAT)
        reward = self.model.Cast(_reward, 'reward', to=core.DataType.FLOAT)
        label = self.model.Cast(_label, 'label', to=core.DataType.INT32)

        feature = self.model.StopGradient(feature, feature)
        expect = self.model.StopGradient(expect, expect)
        reward = self.model.StopGradient(reward, reward)
        label = self.model.StopGradient(label, label)
        return feature, expect, reward, label

    @check_workspace
    def add_conv_block(self, data_in, dim_in, dim_out, kernel=3, with_relu=True):
        pad = (kernel - 1) / 2

        conv = brew.conv(
            self.model,
            data_in,
            'conv_{}'.format(self.conv_count),
            dim_in=dim_in,
            dim_out=dim_out,
            kernel=kernel,
            pad=pad,
            )
        norm = brew.spatial_bn(
            self.model,
            conv,
            'norm_{}'.format(self.conv_count),
            dim_in=dim_out
        )
        if with_relu:
            conv_out = brew.relu(
                self.model,
                norm,
                'relu_{}'.format(self.conv_count),
            )
        else:
            conv_out = norm

        self.conv_count += 1
        return conv_out

    @check_workspace
    def add_res_block(self, data_in):
        conv_out_1 = self.add_conv_block(data_in, self.filter_num, self.filter_num)
        conv_out_2 = self.add_conv_block(conv_out_1, self.filter_num, self.filter_num, with_relu=False)
        res_highway = self.model.Add(
            [conv_out_2, data_in],
            'res_highway_{}'.format(self.res_count)
            )
        res_out = brew.relu(
            self.model,
            res_highway,
            'res_relu_{}'.format(self.res_count),
        )
        self.res_count += 1
        return res_out

    @check_workspace
    def add_policy_head(self, data_in):
        ph_conv_out = self.add_conv_block(data_in, self.filter_num, 2, 1)
        ph_fc = brew.fc(
            self.model,
            ph_conv_out,
            'ph_fc',
            dim_in=2 * self.board_full_size,
            # dim_out=self.board_full_size + 1
            dim_out=self.board_full_size
            )
        predict = brew.softmax(
            self.model,
            ph_fc,
            'predict'
        )
        return predict

    @check_workspace
    def add_value_head(self, data_in):
        vh_conv_out = self.add_conv_block(data_in, self.filter_num, 1, 1)
        vh_fc = brew.fc(
            self.model,
            vh_conv_out,
            'vh_fc',
            dim_in=self.board_full_size,
            dim_out=self.filter_num
        )
        vh_relu = brew.relu(
            self.model,
            vh_fc,
            'vh_relu'
        )
        vh_fc_2 = brew.fc(
            self.model,
            vh_relu,
            'vh_fc_2',
            dim_in=self.filter_num,
            dim_out=1
        )
        value = brew.tanh(
            self.model,
            vh_fc_2,
            'value',
        )
        return value

    @check_workspace
    def add_accuracy(self, predict, label):
        accuracy = brew.accuracy(self.model, [predict, label], 'accuracy')
        return accuracy

    @check_workspace
    def add_train_op(self, predict, expect, value, reward):
        _, xent = self.model.SoftmaxWithLoss([predict, expect], ['_', 'xent'], label_prob=1)
        l2_dis = self.model.SquaredL2Distance([value, reward], 'l2_dis')
        msqrl2 = self.model.AveragedLoss(l2_dis, 'msqrl2')
        loss = self.model.Add([msqrl2, xent], 'loss')
        self.model.AddGradientOperators([loss])

        ITER = brew.iter(self.model, "iter")
        LR = self.model.LearningRate(
            ITER,
            "LR",
            base_lr=self.base_lr,
            policy=self.lr_policy,
            stepsize=self.stepsize,
            gamma=self.gamma
            )
        ONE = self.model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
        # Now, for each parameter, we do the gradient updates.
        for param in self.model.params:
            # Note how we get the gradient of each parameter - ModelHelper keeps
            # track of that.
            param_grad = model.param_to_grad[param]
            # The update is a simple weighted sum: param = param + param_grad * LR
            model.WeightedSum([param, ONE, param_grad, LR], param)

    @check_workspace
    def add_booking_op(self):
        self.model.Print('accuracy', [], to_file=1)
        self.model.Print('loss', [], to_file=1)

        # for param in self.model.params:
        #     self.model.Summarize(param, [], to_file=1)
        #     self.model.Summarize(self.model.param_to_grad[param], [], to_file=1)

    @check_workspace
    def create_net(self):
        workspace.RunNetOnce(self.model.param_init_net)
        workspace.CreateNet(self.model.net)

    @check_workspace
    def run(self):
        workspace.RunNet(self.model.net)


class TrainNet(Net):
    WORKSPACE = 'train'

    @check_workspace
    def build(self, db_name, db_type):
        feature, expect, reward, label = self.add_db_input(db_name, db_type)
        res_in = self.add_conv_block(feature, self.feature_channels, self.filter_num)
        for _ in range(self.res_block_num):
            res_out = self.add_res_block(res_in)
            res_in = res_out
        predict = self.add_policy_head(res_out)
        value = self.add_value_head(res_out)
        self.add_accuracy()
        self.add_train_op(predict, expect, value, reward)
        self.add_booking_op()

        self.create_net()


class TestNet(Net):
    WORKSPACE = 'test'

    @check_workspace
    def build(self, db_name, db_type):
        feature, expect, reward = self.add_db_input(db_name, db_type)
        res_in = self.add_conv_block(feature, self.feature_channels, self.filter_num)
        for _ in range(self.res_block_num):
            res_out = self.add_res_block(res_in)
            res_in = res_out
        predict = self.add_policy_head(res_out)
        value = self.add_value_head(res_out)
        accuracy = self.add_accuracy(predict, label)

        self.create_net()


class DeployNet(Net):
    WORKSPACE = 'deploy'

    def __init__(self, name, board_size):
        super(DeployNet, self).__init__(name, 1, board_size)

    @check_workspace
    def build(self):
        if not workspace.HasBlob('feature'):
            workspace.CreateBlob('feature')
        feature = self.model.StopGradient('feature', 'feature')
        res_in = self.add_conv_block(feature, utils.FEATURE_CHANNEL, utils.FILTER_NUM)
        for _ in range(utils.RES_BLOCK_NUM):
            res_out = self.add_res_block(res_in)
            res_in = res_out
        self.add_policy_head(res_out)
        self.add_value_head(res_out)

        self.create_net()

    @check_workspace
    def get_predict_and_value(self, feature):
        # feature = feature.reshape([1].extend(feature.shape)).astype(np.float32)
        workspace.FeedBlob('feature', feature)
        self.run()
        predict, value = workspace.FetchBlobs(['predict', 'value'])
        return predict[0], value[0, 0]

def main():
    import time
    times = 20
    net = DeployNet("deploy", utils.SIZE)
    net.build()
    f = np.random.sample((1, utils.FEATURE_CHANNEL, utils.SIZE, utils.SIZE))
    st = time.time()
    for _ in range(times):
        net.get_predict_and_value(f)
    et = time.time()
    print("{}".format((et - st) / times))

if __name__ == '__main__':
    main()
