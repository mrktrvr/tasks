'''
test_preprocessing.py
'''
import unittest

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

cdir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(cdir, '..', 'hec')
sys.path.append(lib_dir)

import model.preprocessing as pp
from utils.logger import logger
logger.setLevel('CRITICAL')


class TestPreprocessing(unittest.TestCase):
    def test_sprit_train_test(self):
        df_src = dummy_data()
        drop_idx = df_src.index[0]
        df_src = df_src.drop(index=drop_idx)
        train, test = pp.split_df_train_test(df_src, 7, 0.7)
        self.assertEqual(train.shape, (6552, 1))
        self.assertEqual(test.shape, (2976, 1))
        self.assertEqual(train.index[0].hour, 0)
        self.assertEqual(train.index[-1].hour, 23)
        self.assertEqual(test.index[0].hour, 0)
        self.assertEqual(test.index[-1].hour, 23)

        df_src = dummy_data()
        drop_idx = df_src.index[0]
        df_src = df_src.drop(index=drop_idx)
        drop_idx = df_src.index[10]
        df_src = df_src.drop(index=drop_idx)
        train, test = pp.split_df_train_test(df_src, 7, 0.7)
        self.assertEqual(train.shape, (6552, 1))
        self.assertEqual(test.shape, (2976, 1))
        self.assertEqual(train.index[0].hour, 0)
        self.assertEqual(train.index[-1].hour, 23)
        self.assertEqual(test.index[0].hour, 0)
        self.assertEqual(test.index[-1].hour, 23)

        df_src = dummy_data()
        train, test = pp.split_df_train_test(df_src, 7, 0.5)
        self.assertEqual(train.index[0].hour, 0)
        self.assertEqual(train.index[-1].hour, 23)
        self.assertEqual(test.index[0].hour, 0)
        self.assertEqual(test.index[-1].hour, 23)
        self.assertEqual(train.shape, (4704, 1))
        self.assertEqual(test.shape, (4848, 1))

        drop_idx = df_src.index[0]
        df_src = df_src.drop(index=drop_idx)
        train, test = pp.split_df_train_test(df_src, 7, 0.5)
        self.assertEqual(train.index[0].hour, 0)
        self.assertEqual(train.index[-1].hour, 23)
        self.assertEqual(test.index[0].hour, 0)
        self.assertEqual(test.index[-1].hour, 23)
        self.assertEqual(train.shape, (4704, 1))
        self.assertEqual(test.shape, (4824, 1))

    def test_split_df_feat_tar(self):
        df_src = dummy_data()
        df_feat1, df_feat2, df_tars = pp.split_df_feats_and_tars(df_src)
        self.assertEqual(df_feat1.shape, (390, 168))
        self.assertEqual(df_feat2.shape, (390, 167))
        self.assertEqual(df_tars.shape, (390, 24))

    def test_df_min_max_norm(self):
        file_name = 'test.pkl'
        if os.path.exists(file_name):
            os.remove(file_name)

        df_src = dummy_data()

        df_scaled1 = pp.df_min_max_norm(df_src, file_name, False)
        self.assertAlmostEqual(df_scaled1.max().values[0], 1)
        self.assertAlmostEqual(df_scaled1.min().values[0], 0)
        self.assertNotEqual(df_scaled1.max().values[0], df_src.max().values[0])
        self.assertNotEqual(df_scaled1.min().values[0], df_src.min().values[0])

        df_inversed1 = pp.df_inversed_min_max_norm(df_scaled1, file_name)
        self.assertTrue(np.allclose(df_src, df_inversed1))

        df_scaled2 = pp.df_min_max_norm(df_src, file_name, False)
        self.assertTrue(np.allclose(df_scaled1, df_scaled2))

        df_inversed2 = pp.df_inversed_min_max_norm(df_scaled2, file_name)
        self.assertTrue(np.allclose(df_src, df_inversed2))

        df_scaled3 = pp.df_min_max_norm(df_src, file_name, True)
        self.assertTrue(np.allclose(df_scaled1, df_scaled3))

        df_inversed3 = pp.df_inversed_min_max_norm(df_scaled3, file_name)
        self.assertTrue(np.allclose(df_src, df_inversed3))

        if os.path.exists(file_name):
            os.remove(file_name)


    def test_min_max_norm(self):
        '''
        test of min-max normalisation
        '''
        src1 = np.array([-1, 2, 3, 4, 5, 10, 20])

        min_max_norm = pp.MinMaxNorm('test.pkl')
        dst1 = min_max_norm.fit_transform(src1)
        self.assertAlmostEqual(dst1.max(), 1.0)
        self.assertAlmostEqual(dst1.min(), 0.0)
        self.assertEqual(min_max_norm.param(), (src1.min(), src1.max()))
        dst2 = min_max_norm.transform(src1)
        self.assertTrue(np.allclose(dst1, dst2))

        # --- compare with sklearn module
        sk_min_max_norm = MinMaxScaler()
        dst3 = sk_min_max_norm.fit_transform(src1[:, np.newaxis])[:, 0]
        self.assertTrue(np.allclose(dst1, dst3))
        self.assertTrue(np.allclose(dst2, dst3))

        # --- loading test
        min_max_norm = pp.MinMaxNorm(param_file_name='exception_test.pkl')
        with self.assertRaises(Exception):
            min_max_norm.load_param()

        # --- save and load test
        param_file_name = 'min_max_param.pkl'
        if os.path.exists(param_file_name):
            os.remove(param_file_name)
        min_max_norm = pp.MinMaxNorm(param_file_name=param_file_name)
        dst1 = min_max_norm.fit_transform(src1)
        min_max_norm.save_param()
        min_max_norm = pp.MinMaxNorm(param_file_name=param_file_name)
        min_max_norm.load_param()
        self.assertEqual(min_max_norm.param(), (src1.min(), src1.max()))

        # -- compare with results from same input
        dst2 = min_max_norm.transform(src1)
        self.assertTrue(np.allclose(dst1, dst2))

        # -- compare with results from differnt input
        src2 = np.array([-1, 2, 3, 4, 5, 10, 20]) - 1
        dst3 = min_max_norm.transform(src2)
        self.assertFalse(np.allclose(dst1, dst3))
        self.assertNotEqual(min_max_norm.param(), (src2.min(), src2.max()))
        os.remove(param_file_name)

        # -- compare with results from differnt input, one value
        dst4 = min_max_norm.transform(src2[0])
        self.assertFalse(np.allclose(dst1[0], dst4[0]))
        self.assertFalse(np.allclose(dst2[0], dst4[0]))
        self.assertTrue(np.allclose(dst3[0], dst4[0]))
        self.assertNotEqual(min_max_norm.param(), (src2.min(), src2.max()))

        if os.path.exists(param_file_name):
            os.remove(param_file_name)

if __name__ == '__main__':
    from dummy_data import dummy_data
    unittest.main()
else:
    from .dummy_data import dummy_data
