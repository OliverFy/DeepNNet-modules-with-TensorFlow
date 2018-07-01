#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
from functools import reduce

from . import config
from . import deep_nnet_model


def train(x, y, hidden_neuron_list, ckpt=None, alpha=1.0, keep_prob=1.0, n_train=10000, max_stop_cnt=100, random_seed=None):
    """
    Paramters
    ---------
    x: matrix data. like pandas.DataFrame or numpy.array
    y: matrix data. like pandas.DataFrame or numpy.array
    hidden_neuron_list: list
        各層のノードの数をリストで指定 (nlist>2)
    ckpt: str
        学習結果の保存名．None なら保存されない．
        ex)　'./learned_model/this_model_name.ckpt'
    alpha: float
        loss関数のweightの正則化項の係数。大きいほど、強い正則化になる。
    keep_prob: float
        drop_outに参照される確率。keep_prob=1.0でdrop_outは発生しない。
    n_train: int
        学習回数。現状、batch学習などには対応してないので、全データを指定回数だけ学習する。
    max_stop_cnt: int
        lossの悪化回数をカウントし、指定の回数以上悪化したタイミングで停止する。
    random_seed: int
        乱数シード値の指定です。Noneで指定なし。
    """

    if random_seed:
        tf.set_random_seed(random_seed)

    tf.reset_default_graph()

    inp_num = x.shape[1]
    out_num = y.shape[1]

    x_ph = tf.placeholder(tf.float32, [None, inp_num], name=config.NAME_INPUT_PLACEHOLDER)
    y_ph = tf.placeholder(tf.float32, [None, out_num], name=config.NAME_OUTPUT_PLACEHOLDER)

    # Create model
    predicted_y, weights, activations, biases = deep_nnet_model.create_deep_nnet(x_ph, y_ph, hidden_neuron_list, keep_prob=keep_prob, random_seed=random_seed)

    # Create loss & optimizer
    w_d = sum(map(tf.nn.l2_loss, weights))
    loss = tf.nn.l2_loss(tf.subtract(predicted_y, y_ph)) + alpha*w_d
    train_step = tf.train.AdamOptimizer().minimize(loss)

    for a in activations:
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, a)
    for w in weights:
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
    for b in biases:
        tf.add_to_collection(tf.GraphKeys.BIASES, b)

    # Train
    stop_cnt = 0
    before_loss_val = sys.maxsize
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for step in range(n_train):
            # stop_cntがmax_stop_cntより大きくなると中断
            if stop_cnt > max_stop_cnt:
                break

            feed_dict = {x_ph:x, y_ph:y}
            sess.run(train_step, feed_dict=feed_dict)

            # loss の悪化があるとstop_cntに足し上げ。
            loss_val = sess.run(loss, feed_dict=feed_dict)
            if before_loss_val <= loss_val:
                stop_cnt += 1
            before_loss_val = loss_val

            # 100回に一回モニタリングをする。
            if step%100==0:
                print('step: {},\loss: {}'.format(step, sess.run(loss, feed_dict=feed_dict)))

        if ckpt:
            saver.save(sess, ckpt)
