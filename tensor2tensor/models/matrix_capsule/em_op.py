#encoding=utf-8
#__author__=veinpy

"""
The EM operation for Capsule Network

mostly referenced from Maxtrix-Capsules-EM-Tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

class EM_op():
    """
    EM routing operation

    """
    def __init__(self, hparams):
        self.hparams = hparams

    def E_op(self, hparams,**iter_params):
        iters = iter_params['iters']
        votes_in = iter_params['votes_in']
        #batch_sizez = votes_in.get_shape()[0]
        activation_in = iter_params['activation_in']
        nchannel_output = iter_params['nchannel_output']
        sigma_square = iter_params["sigma_square"]
        miu = iter_params['miu']
        activation_out = iter_params["activation_out"]

        if iters ==0 :
            tmptensor = tf.ones_like(activation_in[:,:,0],dtype=tf.float32)
            tmptensor = tf.expand_dims(tmptensor, -1)
            tmptensor = tf.tile(tmptensor, [1,1, nchannel_output])
            r = tmptensor / nchannel_output

        else:
            # Contributor: Yunzhi Shi
            # log and exp here provide higher numerical stability especially for bigger number of iterations
            log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - (tf.square(votes_in - miu) / (2*sigma_square))
            log_p_c_h = log_p_c_h - (tf.reduce_max(log_p_c_h, axis=[2,3], keep_dims=True) - tf.log(10.))

            p_c = tf.exp(tf.reduce_sum( log_p_c_h, axis=3))

            ap = p_c * tf.reshape(activation_out, shape=[-1, 1 , nchannel_output])

            r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + hparams.epsilon)

        return r

    def M_op(self, hparams, **iter_params):
        iters = iter_params['iters']
        iter_routing = iter_params['iter_routing']
        r = iter_params['r']
        activation_in = iter_params['activation_in']
        nchannel_input = iter_params['nchannel_input']
        nchannel_output = iter_params['nchannel_output']
        votes_in = iter_params["votes_in"]
        n_features = iter_params["n_features"]
        beta_v = iter_params['beta_v']
        beta_a = iter_params['beta_a']
        r = r * activation_in
        r = r / (tf.reduce_sum(r, axis=2, keep_dims=True) + hparams.epsilon)

        r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
        r1 = tf.reshape(r / (r_sum + hparams.epsilon), shape=[-1, nchannel_input, nchannel_output, 1])

        miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
        sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                     axis=1, keep_dims=True) + hparams.epsilon

        if iters == iter_routing -1:
            r_sum = tf.reshape(r_sum, [-1, nchannel_output, 1])
            cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                            shape=[-1, nchannel_output, n_features])))) * r_sum

            activation_out = tf.nn.softmax(hparams.ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))

        else:
            activation_out = tf.nn.softmax(r_sum)

        iter_params.update({"miu": miu})
        iter_params.update({"sigma_square": sigma_square})
        iter_params.update({"activation_out": activation_out})

        return miu, activation_out, iter_params

    def routing(self, hparams,**params):
        # for now, routing is based on exists algorithm
        # but the params inputed can be more fleasible
        votes = params['votes']
        activation = params['activation']
        nchannel_output = params['nchannel_output']
        regularizer = params['regularizer'] if 'regularizer' in params else tf.contrib.layers.l2_regularizer(5e-04)
        iter_routing = params['iter_routing'] if "iter_routing" in params else self.hparams.iter_routing
        assert iter_routing

        tag  = params['tag'] if 'tag' in params else False

        test = []
        nchannel_input = int(activation.get_shape()[1])
        n_features = int(votes.get_shape()[-1])

        sigma_square = []
        miu = []
        activation_out = []

        beta_v = tf.get_variable("beta_v", shape=[nchannel_output, n_features], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                 regularizer=regularizer
                                 )
        beta_a = tf.get_variable("beta_a", shape=[nchannel_output], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                                 regularizer=regularizer
                                 )

        # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
        # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
        votes_in = votes
        activation_in = activation

        routing_params = {}
        routing_params.update({"tag": tag})
        routing_params.update({"nchannel_input": nchannel_input})
        routing_params.update({"n_features":n_features})
        routing_params.update({"sigma_square":sigma_square})
        routing_params.update({"miu":miu})
        routing_params.update({"activation_out":activation_out})
        routing_params.update({"beta_v":beta_v})
        routing_params.update({"beta_a":beta_a})
        routing_params.update({"votes_in":votes_in})
        routing_params.update({"activation_in":activation_in})
        routing_params.update({"nchannel_output": nchannel_output})
        routing_params.update({"iter_routing": iter_routing})
        for iters in range(iter_routing):
            routing_params.update({"iters": iters})
            # e_step
            r = self.E_op(hparams, **routing_params)
            routing_params.update({'r': r})

            # m_step
            miu, activation_out, routing_params = self.M_op(hparams, **routing_params)
        return miu, activation_out