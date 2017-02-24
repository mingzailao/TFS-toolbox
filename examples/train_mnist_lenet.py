from tfs.models.lenet import LeNet
import tensorflow as tf
net = LeNet()
print net
net.build([None,28,28,1])

