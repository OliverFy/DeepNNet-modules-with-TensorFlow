#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済みのグラフから、LRP(Layer-wise Relevance Propagation)を取得する。
"""
from copy import deepcopy

import numpy as np


from .deep_nnet_graph_info import get_activations
from .deep_nnet_graph_info import get_weights

def get_LRP(x, ckpt, method=None):
    val_activations = get_activations(x, ckpt)
    val_weights =  get_weights(ckpt)

    val_activations = list(reversed(val_activations))
    val_weights = list(reversed(val_weights))

    results = []
    for i in range(x.shape[0]):
        results.append(_get_LRP(val_activations, val_weights, sample_index=i, method=method))

    return np.array(results)

def _get_LRP(val_activations, val_weights, sample_index=0, eps=1e-10, method=None): #☆中川さんに確認
    # sample_index=0
    for n_layer in range(len(val_activations)):
        ## n_layer=2
        if n_layer==0:
            x_i = val_activations[n_layer][sample_index]
            R_i = x_i
        else:
            R_j = deepcopy(R_i)
            x_i = val_activations[n_layer][sample_index]
            w_ij = val_weights[n_layer-1]
            if method in ['max', 'MAX']:
                w_ij[w_ij<0] = 0.0
            z_ij = w_ij * x_i.reshape(-1,1)
            normalized_z_ij = z_ij / (z_ij.sum(axis=0) + eps)
            R_i = np.sum(normalized_z_ij * R_j, axis=1)
    return R_i
