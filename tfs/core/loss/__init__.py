# -*- coding: utf-8 -*-

import tensorflow as tf




def mean_squared_error(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred),reduction_indices = 1)

def mean_absolute_error(y_true,y_pred):
    return tf.reduce_mean(tf.abs(y_true-y_pred),reduction_indices = 1)

def categorical_crossentropy(y_true, y_pred):
    try:
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_pred,
                                                       logits=y_true,)
    except TypeError:
        return tf.nn.softmax_cross_entropy_with_logits(y_true,
                                                       y_pred)

def binary_crossentropy(y_true, y_pred):
    try:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_pred,
                                                                      logits=y_true))
    except TypeError:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_true,
                                                                      y_pred))


loss_table={
    'mse': mean_squared_error,
    'mas': mean_absolute_error,
    'categorical_crossentropy': categorical_crossentropy,
    'binary_crossentropy': binary_crossentropy
}
