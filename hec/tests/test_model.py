'''
test_model.py
'''
import unittest

import os
import sys
import pandas as pd
import numpy as np

cdir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(cdir, '..', 'hec')
sys.path.append(lib_dir)

from utils.logger import logger
from model.model import Model
logger.setLevel('CRITICAL')


class TestModel(unittest.TestCase):
    def test_model(self):
        l_r = Model()
        self.assertTrue(l_r.get_model() is None)

        file_name = 'test.pkl'
        if os.path.exists(file_name):
            os.remove(file_name)
        with self.assertRaises(Exception):
            l_r.load_model(file_name)

        features = np.random.rand(2, 168)
        target = np.random.rand(2, 24)

        # --- fit
        l_r.fit(features, target)
        self.assertTrue(l_r.get_model() is not None)

        # --- predict
        pred = l_r.predict(features)
        self.assertTrue(np.allclose(target, pred))

        # --- save, load and predict
        l_r.save_model(file_name)
        l_r.load_model(file_name)
        self.assertTrue(np.allclose(target, pred))

        # --- different data and shape
        features = np.random.rand(1, 168)
        pred = l_r.predict(features)
        self.assertFalse(np.allclose(target[0], pred))
        self.assertFalse(np.allclose(target[1], pred))

        # --- remove file
        if os.path.exists(file_name):
            os.remove(file_name)


if __name__ == '__main__':
    unittest.main()
