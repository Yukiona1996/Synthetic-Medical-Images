# TensorFlow 1 obsolete

# import math
# import numpy as np 
# import tensorflow as tf

# from tensorflow.python.framework import ops

# from utils import *

# try:
#   image_summary = tf.image_summary
#   scalar_summary = tf.scalar_summary
#   histogram_summary = tf.histogram_summary
#   merge_summary = tf.merge_summary
#   SummaryWriter = tf.train.SummaryWriter
# except:
#   image_summary = tf.summary.image
#   scalar_summary = tf.summary.scalar
#   histogram_summary = tf.summary.histogram
#   # merge_summary = tf.summary.merge
#   # SummaryWriter = tf.summary.FileWriter

# if "concat_v2" in dir(tf):
#   def concat(tensors, axis, *args, **kwargs):
#     return tf.concat_v2(tensors, axis, *args, **kwargs)
# else:
#   def concat(tensors, axis, *args, **kwargs):
#     return tf.concat(tensors, axis, *args, **kwargs)

# class batch_norm(object):
#   def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
#     with tf.variable_scope(name):
#       self.epsilon  = epsilon
#       self.momentum = momentum
#       self.name = name

#   def __call__(self, x, train=True):
#     return tf.contrib.layers.batch_norm(x,
#                       decay=self.momentum, 
#                       updates_collections=None,
#                       epsilon=self.epsilon,
#                       scale=True,
#                       is_training=train,
#                       scope=self.name)
# #concatenate
# def conv_cond_concat(x, y):
#   """Concatenate conditioning vector on feature map axis."""
#   x_shapes = x.get_shape()
#   y_shapes = y.get_shape()
#   return concat([
#     x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

# def conv2d(input_, output_dim, 
#        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#        name="conv2d"):
#   with tf.variable_scope(name):
#     w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#               initializer=tf.truncated_normal_initializer(stddev=stddev))
#     conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

#     biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#     conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

#     return conv

# def deconv2d(input_, output_shape,
#        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#        name="deconv2d", with_w=False):
#   with tf.variable_scope(name):
#     # filter : [height, width, output_channels, in_channels]
#     w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
#               initializer=tf.random_normal_initializer(stddev=stddev))
    
#     try:
#       deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
#                 strides=[1, d_h, d_w, 1])

#     # Support for verisons of TensorFlow before 0.7.0
#     except AttributeError:
#       deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
#                 strides=[1, d_h, d_w, 1])

#     biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#     deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

#     if with_w:
#       return deconv, w, biases
#     else:
#       return deconv

# def lrelu(x, leak=0.2, name="lrelu"):
#   return tf.maximum(x, leak*x)

# def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
#   shape = input_.get_shape().as_list()

#   with tf.variable_scope(scope or "Linear"):
#     matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
#                  tf.random_normal_initializer(stddev=stddev))
#     bias = tf.get_variable("bias", [output_size],
#       initializer=tf.constant_initializer(bias_start))
#     if with_w:
#       return tf.matmul(input_, matrix) + bias, matrix, bias
#     else:
#       return tf.matmul(input_, matrix) + bias

# ------------------------------------------------------------------------------------------------------------------------------------------ #

# TensorFlow 2 updated code

# import math
# import numpy as np
import tensorflow as tf

# Replace deprecated functions
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
# logdir = "/home/lakshmi/Anu/MTech_Capstone/DualGANAndU-Net"
# max_queue = 100
# flush_millis = 10 * 1000  # Flush every 10 seconds (in milliseconds)
# filename_suffix = "events"
# name = "Anu"
# summary_writer = tf.summary.create_file_writer(
#     logdir=logdir,
#     max_queue=max_queue,
#     flush_millis=flush_millis,
#     filename_suffix=filename_suffix,
#     name=name,
#     experimental_trackable=True,
#     experimental_mesh=None
# )

def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, train=True):
        return tf.keras.layers.BatchNormalization(epsilon=self.epsilon, momentum=self.momentum, name=self.name)(x, training=train)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]], dtype=x.dtype)], axis=3)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.name_scope(name):
        w = tf.Variable(tf.random.truncated_normal([k_h, k_w, input_.shape[-1], output_dim], stddev=stddev), name='w')
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.Variable(tf.constant(0.0, shape=[output_dim]), name='biases')
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.shape)

        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.name_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.Variable(tf.random.normal([k_h, k_w, output_shape[-1], input_.shape[-1]], stddev=stddev), name='w')

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.Variable(tf.constant(0.0, shape=[output_shape[-1]]), name='biases')
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.shape)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = tf.shape(input_)

    with tf.name_scope(scope or "Linear"):
        matrix = tf.Variable(tf.random.normal([shape[1], output_size], stddev=stddev), name="Matrix")
        bias = tf.Variable(tf.constant(bias_start, shape=[output_size]), name="bias")
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias