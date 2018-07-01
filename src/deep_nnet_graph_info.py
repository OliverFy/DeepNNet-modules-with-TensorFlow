#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from . import config

def get_weights(ckpt):
    """
    ckptに保存された学習結果networkの重みを取得する
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt)))

        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)

        val_weights = []
        for w in weights:
            val = sess.run(w)
            val_weights.append(val)

    return val_weights

def get_biases(ckpt):
    """
    ckptに保存された学習結果networkのバイアス情報を取得する
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt)))

        biases = tf.get_collection(tf.GraphKeys.BIASES)

        val_biases = []

        for b in biases:
            val = sess.run(b)
            val_biases.append(val)

    return val_biases

def get_activations(x, ckpt):
    """
    ckptに保存された学習結果networkの各レイヤーの活性化結果の値を取得する。
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt)))

        graph = tf.get_default_graph()
        x_ph = graph.get_tensor_by_name(config.NAME_INPUT_PLACEHOLDER+":0")
        feed_dict = {x_ph:x}

        activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)

        val_activations = []

        for a in activations:
            val = sess.run(a, feed_dict=feed_dict)
            val_activations.append(val)

    return val_activations
