# -*- coding: utf-8 -*-

import warnings


class Regularizer(object):
    def __call__(self,x):
        return 0
    def __str__(self):
        return {"name":self.__class__.__name__}

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
        return {
            "name":self.__class__.__name__,
            "l1"  :self._l1,
            "l2"  :self._l2
        }

def l1(l=0.01):
    return L1L2Regularizer(l1=l)

def l2(l=0.01):
    return L1L2Regularizer(l2=l)

def l1l2(l1=0.01,l2=0.01):
    return L1L2Regularizer(l1=l1,l2=l2)

regularizier_func_table={
    "l1":l1,
    "l2":l2,
    "l1l2":l1l2
}
