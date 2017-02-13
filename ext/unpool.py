#
# author:crackhopper
#
# we need unpooling operation to do the network visualization task. like
# described in the paper
# https://scholar.google.com/scholar?q=Zeiler+Visualizing+and+understanding+convolutional+networks
# and it has been implemented by yosinski via utilizing the backward pass in
# caffe framework. https://github.com/yosinski/deep-visualization-toolbox
#
# To implement the max unpooling operator, there is an issue discussing it
# https://github.com/tensorflow/tensorflow/issues/2169. However, as menstioned
# by https://github.com/tensorflow/tensorflow/issues/6035, we can not use those
# code directly because the lack of cpu implementation of
# tf.nn.max_pool_argmax.
#
# The code
# https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py
# also gives some hints, but it is not max unpooling.
#
# as mentioned in the issue by girving
# https://github.com/tensorflow/tensorflow/issues/2169, we can utilize the
# gradient operation for max_unpooling pass. (just like in caffe). So we would
# try to do things in this way.
#

from __future__ import division
import tensorflow as tf
import numpy as np

def max_unpool(indata, origin, origin_pooled):
    in_shape = indata.get_shape().as_list()
    out_shape = origin.get_shape().as_list()
    assert in_shape==origin_pooled.get_shape().as_list()
    assert out_shape[0]==in_shape[0] and out_shape[3]==in_shape[3]
    assert out_shape[1]%in_shape[1]==0 and out_shape[2]%in_shape[2]==0
    assert indata.dtype==origin.dtype

    sh = out_shape[1]//in_shape[1]
    sw = out_shape[2]//in_shape[2]

    fx = tf.expand_dims(tf.reshape(indata, [-1]),-1) # (b*h*w*c)x1
    repl_mat = tf.ones((1,sh*sw)) # 1x(sh*sw)
    prod = tf.matmul(fx, repl_mat)  # (b*h*w*c) x(sh*sw)

    # b x h x w x c x sh x sw
    prod = tf.reshape(prod, [-1, in_shape[1], in_shape[2], in_shape[3], sh, sw])

    # b x h x sh x w x sw x c
    prod = tf.transpose(prod, [0, 1, 4, 2, 5, 3])

    # b x (h*sh) x (w*sw) x c  which equals out_shape
    prod = tf.reshape(prod, [-1, in_shape[1] * sh, in_shape[2] * sw, in_shape[3]])

    # max pooling mask
    mask = tf.gradients(origin_pooled,origin)[0]

    return prod*mask

