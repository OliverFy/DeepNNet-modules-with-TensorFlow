#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from .. import deep_nnet_get_LRP
from .. import deep_nnet_trainer
from .. import deep_nnet_predictor
from .. import deep_nnet_visualizer

"""
import unittest
import numpy as np
from src import deep_nnet_get_LRP
from src import deep_nnet_trainer
from src import deep_nnet_predictor
from src import deep_nnet_visualizer
"""

class TestUM(unittest.TestCase):
 
    def setUp(self):
        pass

    def test_00(self):
        """
        構築したネットワークグラフをtensorboardで可視化できるようにする。
        """
        
        x = np.random.rand(100, 3).astype(np.float32)
        y = np.random.rand(100, 1).astype(np.float32)

        nslist = [5,2]
        ckpt = r'./learned_model/test_01.ckpt'
        
        deep_nnet_trainer.train(x, y, nslist, ckpt=ckpt, n_train=0)
        
        deep_nnet_visualizer.visualize(ckpt)
                
        
 
    def test_01(self):
        """
        学習したネットワークが解けて当たり前を解けるか？
        """
        x = np.random.rand(1000, 10).astype(np.float32)
        y = 1 * x[:,0] + 2 * x[:,1] - 1 * x[:,2] + 5 * x[:,3] * x[:,4]
        y = y.reshape(-1,1)
        
        x_train, y_train = x[:700], y[:700]
        x_test, y_test   = x[700:], y[700:]
        
        # モジュールを使ってネットワークを構築
        nslist = [20, 10, 5]
        ckpt = r'./learned_model/test_01.ckpt'
        
        deep_nnet_trainer.train(x_train, y_train, nslist, ckpt=ckpt, keep_prob=1.0, alpha=1.0, max_stop_cnt=100)
        predicted = deep_nnet_predictor.predict(x_test, ckpt=ckpt)
        
        mae = np.abs(y_test - predicted).mean()
        
        # 比較対象としてLinearRegressionで考える（これは交差項を学習できない）
        from sklearn.linear_model import LinearRegression
        LR = LinearRegression()
        LR.fit(x_train, y_train)
        LR_predicted = LR.predict(x_test)
        
        LR_mae = np.abs(y_test - LR_predicted).mean()
        
        # 平均予測も比較対象としてみる。
        mean_mae = np.abs(y_test - y_train.mean()).mean()
        
        # Assertion
        self.assertGreater(mean_mae, mae)
        self.assertGreater(LR_mae, mae)
            
 
    def test_02(self):
        """
        LRPの値が正常化かを確認する。
        """        
        x = np.random.rand(100, 10)
        y = 2 * x[:,0] + 1 * x[:,1] - 3 * x[:,2] + 5 * x[:,3] * x[:,4]
        y = y.reshape(-1,1)

        nslist = [10,20,10]
        ckpt = r'./logs/test_p10_get_LRP.ckpt'
        
        deep_nnet_trainer.train(x, y, nslist, ckpt=ckpt)    
        Rs = deep_nnet_get_LRP.get_LRP(x, ckpt)
        
        # 正解である変数0-4が特徴的であるかを確認する。
        print(Rs.mean(axis=0))
        print(np.median(Rs, axis=0))
        print(np.std(Rs, axis=0))
        

if __name__ == '__main__':
    unittest.main()



