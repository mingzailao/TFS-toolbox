import numpy as np
import inspect
import types
class Param(object):
  def __str__(self):
    info=[]
    for k in self.__dict__:
      value = self.__dict__[k]
      if isinstance(value,types.FunctionType):
        value = value.__module__ +'.'+ value.__name__
      info.append('%s :  %s'%(k,str(value)))
    return '\n'.join(info)

class Layer(object):
  def __init__(self,*args):
    argnames,_,_,_ = inspect.getargspec(type(self).__init__)
    self.param = Param()
    for k,v in zip(argnames[1:],args):
      self.param.__dict__[k]=v
    self.param.name = self.get_unique_name(self.param.name)
    self.name = self.param.name

  _name_counter=0
  def get_unique_name(self,name):
    if name: return name
    name = str(type(self).__name__)
    Layer._name_counter+=1
    return '%s_%d'%(name,Layer._name_counter)

  def build(self,inTensor):
    '''Construct the layer.'''
    raise NotImplementedError('Must be implemented by the subclass.')

  def inverse(self,outTensor):
    print '%s doesn\'t define inverse op, ignore the layer'% type(self).__name__
    self._inv_in = outTensor
    self._inv_out = outTensor
    return self._inv_out
