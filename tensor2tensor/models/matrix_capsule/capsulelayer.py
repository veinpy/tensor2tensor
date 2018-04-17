# encoding=utf-8
# __author__=veinpy

from tensor2tensor.models.matrix_capsule.kernel_op import *
from tensor2tensor.models.matrix_capsule.Layer import Layer

"""
eacg layer in _NET return:
        output, extra_info
        >> extra_info: dict
        >> output: Tensor
"""
_NET = {'conv': capsule_convolution_2d, 'convPrimary':capsule_convolution_2d_primary,
        'classOutput': classOutput}

class capsuleLayer():
    """
    Matrix_capsule_layer, including:

    """

    def __init__(self,
                 layer,
                 hparams,
                 **netparams
                 ):
        """
        Args:
            layer:
            layersize:
            kernel_size:
            strid:
            padding:
            name:
            layertype: 'conv', 'rnn', '..'
        Returns:
        """
        inputs = layer.outputs
        self.netparams = netparams
        self.hparams = hparams
        layertype = netparams['layertype']
        assert layertype in _NET
        network = _NET[layertype]

        outputs,extra = network(inputs, hparams, **netparams)

        self.outputs = outputs
        self.extra = extra
        #self.all_layers = list(layer.all_layers)
        #self.all_layers.extend(self.outputs)

        """
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        """

    def routing(self):
        pass

    def loss(self):
        pass


class PrimaryCapLayer(Layer):
    """
    Primary Capsule Layer
    implement Matrix Capsule, which contains  Pose and activation
    """
    def __init__(self, inputs=None, name='input'):
        Layer.__init__(self, inputs=inputs, name=name)
        tf.logging.info("InputLayer  %s: %s" % (self.name, inputs.get_shape()))
        self.outputs = inputs
        self.all_layers = []
        self.all_params = []
        self.all_drop = {}


