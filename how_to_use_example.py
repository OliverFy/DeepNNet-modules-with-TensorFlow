#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japanese:
    このモジュールの使い方サンプル.

English?:
    how to use this modules.    
"""

import numpy as np
from src import deep_nnet_get_LRP
from src import deep_nnet_trainer
from src import deep_nnet_predictor

# set up sample dataset.
x = np.random.rand(1000, 10).astype(np.float32)
y = 1 * x[:,0] + 2 * x[:,1] - 1 * x[:,2] + 5 * x[:,3] * x[:,4]
y = y.reshape(-1,1)

# split train and test
x_train, y_train = x[:700], y[:700]
x_test, y_test   = x[700:], y[700:]

# set args to used train. ckpt is file name where save learned model.
nslist = [20, 10, 5]
ckpt = r'./learned_model/test_01.ckpt'

# train
deep_nnet_trainer.train(x_train, y_train, nslist, ckpt=ckpt, keep_prob=1.0, alpha=1.0, max_stop_cnt=100)
# predict on test
predicted = deep_nnet_predictor.predict(x_test, ckpt=ckpt)

# check the accuracy.
mae = np.abs(y_test - predicted).mean()

# get LRP
Rs = deep_nnet_get_LRP.get_LRP(x_train, ckpt)


