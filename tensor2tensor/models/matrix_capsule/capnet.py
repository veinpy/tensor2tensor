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
from tensor2tensor.models.matrix_capsule.capsulelayer import capsuleLayer, PrimaryCapLayer
from tensor2tensor.models.matrix_capsule.utils import *
from tensor2tensor.models.matrix_capsule.Layer import Layer,InputLayer

import tensorflow as tf

@registry.register_model("capsule")
class Capsule_Img(t2t_model.T2TModel):

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

        #outputs = Layer(inputs,name="inputs")
        outputs = InputLayer(inputs)
        #outputs = PrimaryCapLayer(inputs)

        # validate nchannel_output[i] == nchannel_input[i+1]
        for params in hparams.capsuleLayerParams:
            outputs = capsuleLayer(outputs, hparams, **params)



@registry.register_hparams
def capsule_img_base():
    hparams = common_hparams.basic_params1()
    hparams.add_hparam("iter_routing", False)
    hparams.add_hparam("epsilon", 1e-9)
    hparams.add_hparam("ac_lambda0", 0.01)
    layerparams = []
    layerparams.append({"kernel_size":[1,1], "stride":[1,1,1,1],"padding":"VALID","scope":"conv_primary",
                        "layertype":"convPrimary","nchannel_output":8, "nchannel_input":None,"iter_routing":None})

    layerparams.append({'kernel_size':[3,3], "stride":[1,2,2,1],'padding':'VALID','scope':'conv_cap1',
                        "layertype":'conv', 'nchannel_output': 32, 'nchannel_input':8, "iter_routing":10})
    
    layerparams.append({'kernel_size':[3,3], "stride":[1,1,1,1],'padding':'VALID','scope':'conv_cap2',
                        "layertype":'conv', 'nchannel_output': 32, 'nchannel_input':32, "iter_routing":10})

    hparams.add_hparam('netstructure', [8, 16, 16]) # mnist
    hparams.add_hparam("capsuleLayerParams", layerIter(layerparams)
                        )
    """
    ...
    """
    return hparams