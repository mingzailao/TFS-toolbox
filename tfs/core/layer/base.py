import numpy as np
import inspect
import types

easyname_dict={"FullyConnect": "FC",
               "Conv2d":"Conv2d",
               "MaxPool":"MaxPool",
               "LRN":"LRN",
               "Softmax":"Softmax"}

class Param(object):
  def __str__(self):
    info=[]
    for k in self.__dict__:
      value = self.__dict__[k]
      if isinstance(value,types.FunctionType):
        value = value.__module__ +'.'+ value.__name__
      info.append('%s :  %s'%(k,str(value)))
    return '\n'.join(info)

  def copy(self):
    obj = Param()
    obj.__dict__ = self.__dict__.copy()
    return obj

class Layer(object):
  def __init__(self,*args):
    argnames,_,_,_ = inspect.getargspec(type(self).__init__)
    self.param = Param()
    for k,v in zip(argnames[1:],args):
      self.param.__dict__[k]=v
    self.param.name = self.get_unique_name(self.param.name)
    self.name = self.param.name
    self.net = None # it is set by class Network

  _name_counter=0
  def get_unique_name(self,name):
    if name: return name
    name = str(type(self).__name__)
    Layer._name_counter+=1
    return '%s_%d'%(name,Layer._name_counter)

  def build(self,inTensor):
    '''Run the layer. '''
    raise NotImplementedError('Must be implemented by the subclass.')

  def inverse(self,outTensor):
    print '%s doesn\'t define inverse op, ignore the layer'% type(self).__name__
    self._inv_in = outTensor
    self._inv_out = outTensor
    return self._inv_out

  def copyTo(self,to_net):
    cls = type(self)
    obj = cls(**self.param.__dict__)
    obj.net = to_net
    return obj

  def __str__(self):
    """
    The info of this layer
    """
    father_before   ="=={:15}==={:8}=".format(15*"=",8*"=")
    father_attribute="| {:<15} | {:<8} ".format("Name","Type")
    father_value    ="| {:<15} | {:<8} ".format(self.param.name,easyname_dict[type(self).__name__])
    # TODO: add compute_param(self) to the base class ?

    for key in self.param.__dict__.keys():
      if key!='name':
        if key!="activation":
          father_before  +="=={:<8}=".format("="*8)
          father_attribute+="| {:<8} ".format(key)
          father_value    +="| {:<8} ".format(self.param.__dict__[key])
        else:
          if  self.param.__dict__[key]:
            father_before  +="=={:<8}=".format("="*8)
            father_attribute+="| {:<8} ".format('activate')
            father_value    +="| {:<8} ".format(self.param.__dict__[key].func_name)
          else:
            father_before  +="=={:<10}=".format("="*10)
            father_attribute+="| {:<10} ".format(key)
            father_value    +="| {:<10} ".format(self.param.__dict__[key])
    return father_before+"|\n"+father_attribute+"|\n"+father_value+'|'
