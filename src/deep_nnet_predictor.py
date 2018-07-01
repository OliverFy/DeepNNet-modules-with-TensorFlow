#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import os

from . import config

def predict(x, ckpt):
    """
    Paramters
    ---------
    x: matrix data. like pandas.DataFrame or numpy.array
    ckpt: str
        学習結果の保存名．None なら保存されない．
        ex)　'./learned_model/this_model_name.ckpt'
    """
    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt)))

        graph = tf.get_default_graph()
        x_ph = graph.get_tensor_by_name(config.NAME_INPUT_PLACEHOLDER+":0")
        y    = graph.get_tensor_by_name(config.NAME_OUTPUT_LAYER+":0")
        feed_dict = {x_ph:x}
        predicted = sess.run(y, feed_dict=feed_dict)
    return predicted
