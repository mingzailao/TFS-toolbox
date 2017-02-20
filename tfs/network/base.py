import tensorflow as tf
import numpy as np
from tfs.core.layer import func_table

def _layer_function(layerclass):
  def func(self,*args,**kwargs):
    layer = layerclass(*args,**kwargs)
    self.layers.append(layer)
    return self
  return func

def _network_meta(future_class_name, future_class_parents, future_class_attr):
  for k in func_table:
    future_class_attr[k]=_layer_function(func_table[k])
  return type(future_class_name, future_class_parents, future_class_attr)

# the base class of the net
class Network(object):
  __metaclass__ = _network_meta
  def __init__(self,input_shape):
    self._in = tf.placeholder(tf.float32,input_shape)
    self.layers=[]
    self.setup()
    self._out = self.build()

  def setup(self):
    '''Construct the network. '''
    raise NotImplementedError('Must be implemented by the subclass.')

  def build(self):
    tmp = self._in
    for l in self.layers:
      tmp = l.build(tmp)
    self._out = tmp
    return tmp

  # TODO: add the summary function to summary the network
  # TODO: add the param summary
  # TODO: add the clear summary later
  def summary(self):
    """
    The summary of the network
    """
    print "Summary:\n"
    for (i,layer) in  enumerate(self.layers):
      print(" ============================================ ")
      print("| {:10} | {:<10} ".format("Index",i))
      print("| {:10} | {:<10} ".format("Name",layer.param.__dict__['name']))
      for key in layer.param.__dict__.keys():
        if key!='name':
          if key!="activation":
            print("| {:10} | {:<10} ".format(key,layer.param.__dict__[key]))
          else:
            if  layer.param.__dict__[key]:
              print("| {:10} | {:<10} ".format(key,layer.param.__dict__[key].func_name))
            else:
              print("| {:10} | {:<10} ".format(key,layer.param.__dict__[key]))

  def load(self, data_path, session, ignore_missing=False):
    '''Load network weights.
    data_path: The path to the numpy-serialized network weights
    session: The current TensorFlow session
    ignore_missing: If true, serialized weights for missing layers are ignored.
    '''
    data_dict = np.load(data_path).item()
    for op_name in data_dict:
      with tf.variable_scope(op_name, reuse=True):
        for param_name, data in data_dict[op_name].iteritems():
          try:
            var = tf.get_variable(param_name)
            session.run(var.assign(data))
          except ValueError:
            if not ignore_missing:
              raise
