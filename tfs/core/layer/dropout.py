import tensorflow as tf
import numpy as np
import ops
from base import Layer

class Dropout(Layer):
  def __init__(self,
               keep_prob,
               name=None
  ):
    Layer._init(
      self,
      keep_prob,
      name
    )
  def _build(self):
    inTensor = self._in
    output = tf.nn.dropout(inTensor, self.param.keep_prob,
                           name=self.param.name)
    return output

