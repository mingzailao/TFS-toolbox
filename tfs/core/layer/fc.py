import tensorflow as tf
import numpy as np
import ops
from base import Layer
from tfs.core.regularizers import regularizer_list

class FullyConnect(Layer):
  def __init__(self,
               outdim,
               activation = ops.relu,
               name=None,
               W_regularizer=None,
               b_regularizer=None
  ):
    Layer._init(
      self,
      outdim,
      activation,
      name,
      W_regularizer,
      b_regularizer
    )

  def _build(self):
    inTensor = self._in
    input_shape = inTensor.get_shape()
    if input_shape.ndims == 4:
      # The input is spatial. Vectorize it first.
      dim = np.prod(input_shape.as_list()[1:])
      output = tf.reshape(inTensor, [-1,dim])
    else:
      output, dim = (inTensor, input_shape[-1].value)
    weights = self._make_variable('weights', shape=[dim, self.param.outdim])
    biases = self._make_variable('biases', [self.param.outdim])
    output = tf.nn.xw_plus_b(output, weights, biases,name=self.name)
    if self.param.activation:
      output= self.param.activation(output, name=self.name)

    if self.param.W_regularizer is not None or self.param.b_regularizer is not None:
      # check the parameters of W_regularizer and b_regularizer
      if self.param.W_regularizer and type(self.param.W_regularizer) not in regularizer_list:
        raise ValueError("The type of W_regularizer is not in  {}".format(regularizer_list))
      if self.param.b_regularizer and type(self.param.b_regularizer) not in regularizer_list:
        raise ValueError("The type of W_regularizer is not in  {}".format(regularizer_list))
      self.compute_regularization()
    return output

  def compute_regularization(self):
    self._regularization=tf.constant(value=0.,dtype=tf.float32)
    if self.param.W_regularizer:
      self._regularization=tf.add(self._regularization,self.param.W_regularizer(self._variables["weights"]))
    if self.param.b_regularizer:
      self._regularization=tf.add(self._regularization,self.param.b_regularizer(self._variables["bias"]))

  def _inverse(self):
    outTensor = self._inv_in
    name = 'inv_'+self.name
    act = self.param.activation
    if act:
      outTensor = act(outTensor)
    weights = tf.transpose(self._variables['weights'])
    inv_fc = tf.matmul(outTensor,weights)
    shape = self._in.get_shape().as_list()
    shape[0]=-1
    inv_fc = tf.reshape(inv_fc,shape)
    print 'inv_fc '+str(outTensor.get_shape().as_list()) + '->' + str(inv_fc.get_shape().as_list())
    return inv_fc

