#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create model of deep neural network.
"""

import tensorflow as tf
from . import config

def create_deep_nnet(inputs, outputs, hidden_neuron_list, epsilon=1e-10, keep_prob=1.0, random_seed=None):
    """
    ARGUMENTs
    ----------
    inputs [tf.Placeholder]:
        tf.Placeholder which is x.
    outputs [tf.Placeholder]:
        tf.Placeholder which is y.
    hidden_neuron_list [list]:
        list of hidden layer neurons numbers.
        ex) 3-hidden-layer [5,10,3]
    epsilon [float]:
        very small number to be used avoid zero-devide errorself.
    random_seed [int]:
        random_seed

    RETURNs
    ----------
    y, weights, activations, biases:
        those are tensors which are used sess.run() of tensorflow.
    """

    if random_seed:
        tf.set_random_seed(random_seed)

    weights = []
    activations = []
    biases = []

    # Input
    inp = inputs
    activations.append(inp)

    # Hidden layer
    for i, ns in enumerate(hidden_neuron_list):
        inp, w_h, z_h, b_h = create_hlayer(inp, ns, epsilon, i, keep_prob)
        weights.append(w_h)
        activations.append(z_h)
        biases.append(b_h)

    # Output
    outputs_dim = int(outputs.get_shape()[1])
    w_o = tf.Variable(tf.truncated_normal(
            [hidden_neuron_list[-1], outputs_dim],
            stddev=tf.sqrt(2.)/tf.sqrt(float(hidden_neuron_list[-1]))),
            name=config.NAME_WEIGHT.format(From=hidden_neuron_list[-1], To=config.NAME_OUTPUT_LAYER))
    b_o = tf.Variable(tf.zeros(outputs_dim), name=config.NAME_OUTPUT_BIASE)
    y_o = tf.add(tf.matmul(inp, w_o), b_o, name=config.NAME_OUTPUT_LAYER)
    weights.append(w_o)
    activations.append(y_o)
    biases.append(b_o)

    return y_o, weights, activations, biases


def create_hlayer(inp, ns, epsilon, idx, keep_prob):
    inp_ns = int(inp.shape[1])

    with tf.name_scope('hlayer_{}'.format(idx)):
        W = tf.Variable(
                tf.truncated_normal(shape=[inp_ns, ns], stddev=1.0/tf.sqrt(float(inp_ns)),
                name='w_h_{}'.format(idx)))

        b = tf.Variable(tf.zeros([ns]), name=config.NAME_HIDDEN_BIASE.format(layerID=idx))
        z = tf.add(tf.matmul(inp, W), b, name=config.NAME_HIDDEN_LAYER.format(layerID=idx))
        h = tf.nn.relu(z)

        keep_prob = tf.constant(keep_prob, tf.float32)

        return tf.nn.dropout(h, keep_prob), W, h, b
