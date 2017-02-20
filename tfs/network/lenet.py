import tensorflow as tf
import numpy as np

from base import Network
class LeNet(Network):
  def setup(self,in_shape):
    (self.conv2d([5,5,in_shape[-1],10],[1,1,1,1])
     .conv2d([5,5,10,10],[1,1,1,1])
     .fc(256)
     .fc(10, activation=None)
    )
