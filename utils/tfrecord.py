import tensorflow as tf
import utils

def generate_example(feature, expect, reward):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.tostring()])),
                'expect': tf.train.Feature(bytes_list=tf.train.BytesList(value=[expect.tostring()])),
                'reward': tf.train.Feature(int64_list=tf.train.Int64List(value=[reward]))
            }
        )
    )

def generate_writer(path):
    return tf.python_io.TFRecordWriter(path)

def generate_dataset(files_list, batch_size):
    dataset = tf.contrib.data.TFRecordDataset(files_list)
    dataset = dataset.map(parse_func)
    dataset = dataset.shuffle(utils.TRAIN_EPOCH_GAME_NUM * utils.SIZE * utils.SIZE)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    return iterator, next_batch

def parse_func(content):
    example = tf.parse_single_example(
        content,
        features={
            'feature': tf.FixedLenFeature([], tf.string),
            'expect': tf.FixedLenFeature([], tf.string),
            'reward': tf.FixedLenFeature([], tf.int64)
            }
        )
    feature = tf.cast(tf.reshape(
        tf.decode_raw(example['feature'], tf.int8),
        (utils.SIZE, utils.SIZE, utils.FEATURE_CHANNEL)
        ), tf.float32)
    expect = tf.squeeze(tf.decode_raw(example['expect'], tf.float32))
    reward = tf.cast(example['reward'], tf.float32)
    return feature, expect, reward
