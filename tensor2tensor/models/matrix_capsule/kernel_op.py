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

from tensor2tensor.models.matrix_capsule.em_op import EM_op

def capsule_convolution_2d(inputs, hparams, **netparams):
    scope = netparams['scope']
    kernel_size = netparams['kernel_size']
    stride = netparams['stride']
    padding = netparams['padding']
    nchannel_output = netparams['nchannel_output']
    nchannel_intput = netparams['nchannel_intput']
    EM = EM_op(hparams)

    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    if scope:
        with tf.variable_scope(scope):
            output = kernel_tile(inputs, kernel_size, stride, padding)
            kernel_tile_shape = output.get_shape()
            # reshape output, split into: pose_tensor and activation_tensor
            # []
            output = tf.reshape(output, shape = [-1, kernel_tile_shape[1]*kernel_tile_shape[2], np.prod(kernel_size)*channel_output, 17])
            activation = tf.reshape(output[:, :, 16], shape=[
                                    -1, np.prod(kernel_size)*nchannel_intput, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:,:,:16], nchannel_output, weights_regularizer, tag=True)

            with tf.variable_scope("routing") as scope:
                routing_params = {'votes': votes, "activation": activation,
                                  "nchannel_output": nchannel_output, 'regularizer': weights_regularizer}
                miu, activation = EM.routing(hparams, **routing_params)
            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.C, 16])
    else:
        pass

def mat_transform(inputs, caps_num_c, regularizer, tag=False):
    caps_num_i = int(inputs.get_shape()[1])
    output = tf.reshape(inputs, shape=[-1, caps_num_i, 1, 4 , 4])  # batch_size * (K*K*pre_layer_nchannal_output) * 16
    # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
    # it has no relationship with the absolute values of w and votes
    # using weights with bigger stddev helps numerical stability
    w = tf.get_variable('w', shape = [1, caps_num_i, caps_num_c, 4,4], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                        regularizer=regularizer)
    w = tf.tile(w, [-1, ])
    output = tf.tile(output, [1,1,caps_num_c, 1, 1])
    votes = tf.reshape(tf.matmul(output, w) , [-1, caps_num_i, caps_num_c,16])
    return votes

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
        stride:  tuple, length = 4
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