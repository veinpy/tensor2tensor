#encoding=utf-8
#__author__=veinpy

"""
capsule network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from tensor2tensor.models.matrix_capsule.kernel_op import *
from tensor2tensor.models.matrix_capsule.capsulelayer import capsuleLayer
from tensor2tensor.models.matrix_capsule.utils import *

import tensorflow as tf

@registry.register_model("capsule")
class Capsule_Img(t2t_model):

    def body(self, features):

        # after input modality,
        # inputs variable is capsule layer
        # base capsule's structure:
        #      4*4 pose matrix,
        #      1 activation scalar
        # one standard capsule layer:
        #      height * weight * (pose + activation)
        # basic operation for capsule layer:
        #      convolution
        #      matrix multiplication
        #      em_routing
        inputs = features['inputs']
        targets = features['targets']

        hparams = self.hparams

        outputs = inputs

        # validate nchannel_output[i] == nchannel_input[i+1]
        abc

        for params in hparams.capsuleLayerParams:
            outputs = capsuleLayer(outputs, **params)



@registry.register_hparams
def capsule_img_base():
    hparams = common_hparams.basic_params1()

    layerparams = []
    layerparams.append({'hidden': 16,'kernel_size':[3,3], "stride":[2],'padding':'VALID', 'scope':'conv_cap1',
                        "layertype":'conv', 'nchannel_output': 32, 'nchannel_input':32})
    layerparams.append({'hidden': 16,'kernel_size':[3,3], "stride":[1],'padding':'VALID', 'scope':'conv_cap1',
                        "layertype":'conv', 'nchannel_output': 32, 'nchannel_input':32})

    hparams.add_hparams('netstructure', [8, 16, 16]) # mnist
    hparams.add_hparams("capsuleLayerParams", layerIter(layerparams)
                        )
    """
    ...
    """
    return hparams