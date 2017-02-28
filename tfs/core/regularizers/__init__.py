# -*- coding: utf-8 -*-

import warnings
import tensorflow as tf

class Regularizer(object):
    def __call__(self,x):
        return 0
    def __str__(self):
        return str({"name":self.__class__.__name__})

class L1L2Regularizer(Regularizer):
    def __init__(self,l1=0.,l2=0.):
        self._l1=l1
        self._l2=l2
    def __call__(self,x):
        regularization=0
        if self._l1:
            regularization+=tf.reduce_sum(self._l1*tf.abs(x))
        if self._l2:
            regularization+=tf.reduce_sum(self._l2*tf.square(x))
        return regularization
    def __str__(self):
        return str({
            "name":self.__class__.__name__,
            "l1"  :self._l1,
            "l2"  :self._l2
        })

regularizer_list=(
    L1L2Regularizer,
)
