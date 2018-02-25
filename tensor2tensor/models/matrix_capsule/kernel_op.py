#encoding=utf-8
#__author__=shimo

"""
capsule operation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def capsule_convolution_2d(inputs, **netparams):
    kernel_size = netparams['kernel_size']
    stride = netparams['stride']
    padding = netparams['padding']



def squash_op(capsules):
    """
    calculate the existance probability for capsule
    v_j = ||s_j||**2 / (1+||s_j||**2) * (s_j / ||s_j||)

    Returns:

    """
    norm_2 =  tf.reduce_sum(tf.square(capsules), axis=-1, keep_dims=True)
    # another choice for stability:
    # norm = tf.sqrt(norm_2 + epislon)
    norm = tf.sqrt(norm_2)
    output = norm_2 / (1+norm_2) * (capsules / norm)
    return output

