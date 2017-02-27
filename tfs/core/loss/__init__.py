# -*- coding: utf-8 -*
import tensorflow as tf


_EPSILON = 10e-8

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x



def mean_squared_error(y_true,y_pred):
    return tf.reduce_mean(input_tensor= tf.square(y_pred-y_true),axis=0,name= "mean_squared_error_operation" )

def mean_absolute_error(y_true,y_pred):
    return tf.reduce_mean(input_tensor= tf.abs(y_pred-y_true),axis=0,name="mean_absolute_error_operation")

def categorical_crossentropy(y_true,y_pred,from_logits=False):
    if not from_logits:
        y_pred/=tf.reduce_sum(y_pred,reduction_indices=len(y_pred.get_shape())-1,keep_dims=True)
        epsilon=_to_tensor(_EPSILON,y_pred.dtype.base_type)
        y_pred =tf.clip_by_value(y_pred,epsilon,1-epsilon)
        return -tf.reduce_sum(y_true* tf.log(y_pred),reduction_indices=len(y_pred.get_shape())-1)
    else:
        try:
            return tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred)
        except TypeError:
            return tf.nn.softmax_cross_entropy_with_logits(y_pred,y_true)

def binary_crossentropy(y_true,y_pred,from_logits=False):
    if not from_logits:
        epsilon=_to_tensor(_EPSILON,y_pred.dtype.base_type)
        y_pred=tf.clip_by_value(y_pred,epsilon,1-epsilon)
        y_pred=tf.log(y_pred/(1-y_pred))
    try:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_true,y_pred),keep_dims=-1)
    except TypeError:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred,y_true),keep_dims=-1)



loss_func_table={
    "mse": mean_absolute_error,
    "mae": mean_absolute_error,
    "binary_crossentropy": binary_crossentropy,
    "categorical_crossentropy": categorical_crossentropy
}
