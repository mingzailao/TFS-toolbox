import tensorflow as tf
import unpool

# TODO: study how to write C++ op, and rewrite the max_unpooling
# import os
# _dir_path = os.path.dirname(os.path.realpath(__file__))
# _zero_out_module = tf.load_op_library(os.path.join(_dir_path,'zero_out.so'))
# zero_out = _zero_out_module.zero_out

max_unpool = unpool.max_unpool
