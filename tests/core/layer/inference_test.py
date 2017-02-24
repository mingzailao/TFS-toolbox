import pytest
import tensorflow as tf
import numpy as np
import tfs.core.layer.ops as ops

from tfs.core.layer.inference import Softmax

@pytest.fixture
def l():
  l = Softmax(
  )
  return l

class TestSoftmax:
  def test_build_inverse(self,l):
    _in = tf.zeros([1,10,10,4])
    _out=l.build(_in)
    assert _out.get_shape().as_list()==[1,10,10,4]

