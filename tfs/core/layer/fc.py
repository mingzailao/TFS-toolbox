import tensorflow as tf
import numpy as np
import ops
from base import Layer

class FullyConnect(Layer):
  def __init__(self,
               outdim,
               activation = ops.relu,
               name=None
  ):
    super(FullyConnect,self).__init__(
      self,
      outdim,
      activation,
      name
    )

  def build(self,inTensor):
    self._in = inTensor
    input_shape = inTensor.get_shape()
    with tf.variable_scope(self.name) as scope:
      if input_shape.ndims == 4:
        # The input is spatial. Vectorize it first.
        dim = np.prod(input_shape.as_list()[1:])
        output = tf.reshape(inTensor, [-1,dim])
      else:
        output, dim = (inTensor, input_shape[-1].value)
      weights = tf.get_variable('weights', shape=[dim, self.param.outdim])
      biases = tf.get_variable('biases', [self.param.outdim])
      output = tf.nn.xw_plus_b(output, weights, biases,name=scope.name)
      if self.param.activation:
        output= self.param.activation(output, name=scope.name)
    self._out = output
    return output

