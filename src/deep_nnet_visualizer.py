#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import os


def visualize(ckpt):
    """
    ckptに保存された学習結果networkをtensorboadで可視化する。
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt)))

        dirname = os.path.dirname(ckpt).replace('./', '')
        tf.summary.FileWriter(dirname, sess.graph)

    print("""
    1) RUN following command on shell.
          >tensorboard --logdir='{dirname}'
    2) visit 'localhost:6006' on any browser.
    """.format(dirname=dirname))
