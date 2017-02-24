from util import *
import tensorflow as tf
# decorators

def run_once_for_each_obj(f):
  """decorate the method which only run once for each object
  """
  def wrapper(self,*args, **kwargs):
    if not hasattr(self,'_has_run'): self._has_run={}
    assert f.__name__ not in self._has_run
    self._has_run[f.__name__] = True
    return f(self,*args, **kwargs)
  wrapper.__name__ = f.__name__
  return wrapper

def local_variable_scope(f):
  """
  """
  def wrapper(self,*args, **kwargs):
    with tf.variable_scope(self.name):
      return f(self,*args, **kwargs)
  wrapper.__name__ = f.__name__
  return wrapper
