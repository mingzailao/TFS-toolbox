import tensorflow as tf
import numpy as np
import ops
from base import Layer

class Softmax(Layer):
  def __init__(self,
               name=None
  ):
    Layer._init(
      self,
      name
    )
  def _build(self):
    inTensor = self._in
    output = tf.nn.softmax(inTensor,name=self.name)
    return output



