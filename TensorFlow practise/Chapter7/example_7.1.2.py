import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist=input_data.read_data_sets("/path/to/mnist/data",dtype=tf.uint8,one_hot=True)
images=mnist.train.images

labels=mnist.train.labels

pixels=images.shape[1]
num_examples=mnist.train.num_examples

