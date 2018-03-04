# encoding=utf-8
# __author__=veinpy

from tensor2tensor.models.matrix_capsule.kernel_op import *

_NET = {'conv': capsule_convolution_2d}

class capsuleLayer():
    """
    Matrix_capsule_layer, including:

    """

    def __init__(self,
                 layer,
                 hparams,
                 scope=None,
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
            scope:
            layertype: 'conv', 'rnn', '..'

        Returns:

        """
        inputs = layer.outputs

        layertype = netparams['layertype']
        assert layertype in _NET
        network = _NET[layertype]

        outputs = network(inputs, hparams, **netparams)

        self.outputs = outputs
        self.all_layers = list(layer.all_layers)
        self.all_layers.extend(self.outputs)

        """
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        """

    def routing(self):
        pass

    def loss(self):
        pass
