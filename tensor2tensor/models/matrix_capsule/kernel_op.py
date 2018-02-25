#encoding=utf-8
#__author__=shimo

"""
capsule operation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import numpy as np

def capsule_convolution_2d(inputs, **netparams):
    scope = netparams['scope']
    kernel_size = netparams['kernel_size']
    stride = netparams['stride']
    padding = netparams['padding']

    if scope:
        with tf.variable_scope(scope):
            output = kernel_tile(inputs, kernel_size, stride, padding)
            # reshape output, split into: pose_tensor and activation_tensor
    else:
        pass

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

def kernel_tile(inputs,  kernel_size, stride, padding):
    """

    Args:
        inputs: Tensor
        kernel_size:
        stride:
        padding:

    Returns:

    """
    tf.logging.info("kernel_size: {}, strides: {}, padding:{}".format(kernel_size, stride, padding))
    input_shape = inputs.get_shape()
    tile_filer_shape = kernel_size + [input_shape[3]] + np.prod(kernel_size)
    tile_filter = np.zeros(tile_filer_shape, dtype=np.float32)
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            tile_filter[i,j,:,i*kernel_size[1]+j] = 1.

    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(inputs, tile_filter_op, strides=stride, padding=padding)
    output_shape = output.get_shape()
    tf.logging.info("kernel tile, output shape is: {}".format(output_shape))

    output = tf.reshape(output, shape= [-1, int(output_shape[1]), int(output_shape[2]), int(input_shape[3]), np.prod(kernel_size)])
    tf.logging.info('kernel_tile, output shape after reshape: {}'.format(output.get_shape()))

    output = tf.transpose(output, perm=[0,1,2,4,3])
    tf.logging.info('kernel_tile, output shape after transpose: {}'.format(output.get_shape()))

    return output