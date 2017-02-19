import tensorflow as tf
import ops
from base import Layer

class Conv2d(Layer):
  def __init__(self,
               ksize,
               strides,
               activation=ops.relu,
               padding='SAME',
               group=1,
               biased=True,
               name=None
  ):
    super(Conv2d,self).__init__(
      self,
      ksize,
      strides,
      activation,
      padding,
      group,
      biased,
      name
    )

  def build(self,inTensor):
    self._in=inTensor
    k_h, k_w, c_i, c_o = self.param.ksize
    group = self.param.group
    strides = self.param.strides
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=self.param.padding)

    with tf.variable_scope(self.name) as scope:
      kernel_shape = [k_h, k_w, c_i / group, c_o]
      kernel = tf.get_variable('weights', shape=kernel_shape)
      if group == 1:
        # This is the common-case. Convolve the input without any further complications.
        output = convolve(self._in, kernel)
      else:
        # Split the input into groups and then convolve each of them independently
        input_groups = tf.split(self._in, group,3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        output = tf.concat(output_groups,3)
      # Add the biases
      if self.param.biased:
        biases_shape = [c_o]
        biases = tf.get_variable('biases', biases_shape)
        output = tf.nn.bias_add(output, biases)
      if self.param.activation:
        output = self.param.activation(output, name=scope.name)
      self.variables = {
        'kernel':kernel,
        'biases':biases
      }
      self._out = output
      return output

