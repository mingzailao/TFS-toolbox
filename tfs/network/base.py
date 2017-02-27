import tensorflow as tf
import numpy as np
from tfs.core.layer import func_table
from tfs.core.util import run_once_for_each_obj
import new
from tfs.core.loss import loss_func_table
#################### NetStructure
def _layer_function(layerclass):
  def func(self,*args,**kwargs):
    layer = layerclass(*args,**kwargs)
    self.append(layer)
    return self
  return func

def _net_sturcture_meta(future_class_name, future_class_parents, future_class_attr):
  for k in func_table:
    future_class_attr[k]=_layer_function(func_table[k])
  return type(future_class_name, future_class_parents, future_class_attr)

class NetStructure(object):
  __metaclass__ = _net_sturcture_meta
  """This class is used for define a network structure by using layers.
  """
  def __init__(self,net,layers=None):
    layers = layers or []
    self.layers=layers
    self.net = net

  def append(self,l):
    self._need_built = True
    l.net = self.net
    self.layers.append(l)

  def __getitem__(self,i):
    return self.layers[i]

  def copy_to(self,net):
    res = NetStructure(net)
    for l in self.layers:
      res.layers.append(l.copy_to(net))
    return res

  def _built_lut(self):
    if not self._need_built: return
    self._lut = {}
    for l in self.layers:
      self._lut[l.name]=l

  def by_name(self,name):
    self._built_lut()
    return self._lut[name]

  def names(self):
    self._built_lut()
    return self._lut.keys()

#################### Network
# decorators
def with_graph(f):
  def with_graph_run(self,*args,**kwargs):
    with self.graph.as_default():
      return f(self,*args,**kwargs)
  # this is important to make the decorator compatiable with run_once_each_obj.
  with_graph_run.__name__=f.__name__
  return with_graph_run

class Network(object):
  def __init__(self):
    self._struct = NetStructure(self)
    self._in = None
    self._out = None
    self._graph = tf.Graph()
    with self.graph.as_default():
      self._sess = tf.Session()

  @property
  def layers(self):
    return self._struct

  @property
  def net_def(self):
    return self._struct

  def __del__(self):
    self.sess.close()

  def setup(self):
    '''Construct the network. '''
    raise NotImplementedError('Must be implemented by the subclass.')

  def setup_with_def(self,struct_def,in_shape=None):
    if isinstance(struct_def,list):
      struct_def = NetStructure(self,layers=struct_def)
    self._struct = struct_def.copy_to(self)
    if in_shape:
      self.build(in_shape)

  def layer_by_name(self,name):
    return self.net_def.by_name(name)

  @property
  def graph(self):
    return self._graph

  @property
  def input(self):
    return self._in

  @property
  def output(self):
    return self._out

  @property
  def sess(self):
    return self._sess

  @with_graph
  @run_once_for_each_obj
  def build(self,input_shape,dtype=tf.float32):
    """Build the computational graph
    inTensor: the network input tensor.
    """
    self._in = tf.placeholder(dtype,input_shape)
    tmp = self._in
    for l in self.layers:
      tmp = l.build(tmp)
    self._out = tmp
    return tmp

  def has_built(self):
    if hasattr(self,'_has_run'):
      if Network.build.__name__ in self._has_run:
        return True
    return False

  def run(self,eval_list,feed_dict):
    return self.sess.run(eval_list, feed_dict=feed_dict)

  @with_graph
  def initialize(self):
    return self.sess.run(tf.global_variables_initializer())

  @with_graph
  def load(self, data_path, ignore_missing=False):
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
            self.sess.run(var.assign(data))
          except ValueError:
            if not ignore_missing:
              raise

  @property
  def in_shape(self):
    if self._in is not None:
      return self._in.get_shape().as_list()
    return None

  def copy(self):
    obj = Network()
    obj.setup_with_def(self.net_def,self.in_shape)
    return obj

  def __str__(self):
    # TODO:
    result=''
    for (i,layer) in enumerate(self.layers):
      result+="Layer {} :\n".format(i)+str(layer)+"\n"
    return result

  def subnet(self,begin_index,end_index):
    obj = Network()
    obj.setup_with_def(self.layers[begin_index:end_index])
    return obj

  @property
  def loss(self):
    return self._loss

  def compile(self,loss_func="categorical_crossentropy"):
    self._loss=0
    self._y_true=tf.placeholder(dtype=tf.float32,shape=self._out.get_shape().as_list())
    assert loss_func in func_table.keys
    self._loss+=func_table[loss_func](self._y_true,self._out)

    for layer in self.layers:
      self._loss+=layer._regularization


  def has_compile(self):
    return hasattr(self._loss)




class CustomNetwork(Network):
  """Automatically called setup and build when construct
  """
  def __init__(self,in_shape=None):
    Network.__init__(self)
    self.default_in_shape = None
    self.setup()
    in_shape = self.default_in_shape
    if not in_shape:
      raise ValueError("must sepecify the default_in_shape attributes, or pass the shape as an argument when construction")

  def setup(self):
    raise NotImplementedError("CustomNetwork Must Implement setup Method")


