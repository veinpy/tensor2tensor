#encoding=utf-8
#__author__=veinpy

"""
t2t modality for capsule net.
which will produce primary capsule
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

import tensorflow as tf

from tensorflow.python.eager import context

@registry.register_image_modality("PrimaryCapsule")
class PrimaryCapsuleModality(modality.Modality):

    """
    primary capsule layer
    """
    def bottom(self, inputs):
        with tf.variable_scope(self.name):
            # inputs will be 3 (RGB pixel) even inputs_shape[3]==1
            inputs = common_layers.standardize_images(inputs)
            if not context.in_eager_mode():
                tf.summary.image("inputs", inputs, max_outputs=2)
