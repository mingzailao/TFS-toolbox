from tfs.core.util import *
class DeconvVisNet(object):
  def __init__(self,netobj):
    self.net = netobj
    self._inv_in = netobj._out
    tmp = netobj._out
    for l in netobj.layers[::-1]:
      tmp = l.inverse(tmp)
    self._inv_out = tmp
    layers = {}
    for l in netobj.layers:
      layers[l.name]=l._out
    self.layers= layers

  def vis_image(self,sess,layer_name,channel_id,image):
    layer_output = sess.run(
      self.layers[layer_name],
      feed_dict={
        self.net._in:image
      })
    to_vis=np.zeros_like(layer_output)
    to_vis[0,...,channel_id]=layer_output[0,...,channel_id]

    generated = sess.run(
      self._inv_out,
      feed_dict={
        self.layers[layer_name]:to_vis,
        self.net._in:image
      })
    gen_img=ensure_uint255(norm01c(generated[0,:],0))
    return gen_img[:,:,::-1]


