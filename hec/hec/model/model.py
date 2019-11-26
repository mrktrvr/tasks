'''
model.py
'''

import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression


cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))

import model.preprocessing as pp
from utils.logger import logger
import utils.data_handler as dh


class Model():

    def __init__(self):
        self._l_r = None

    def fit(self, features: np.ndarray, targets: np.ndarray):
        '''
        fit(features, targets)
        Params
        features: feature vectors
        targets: target vectors
        '''
        logger.info('fit started.')
        logger.info('features size: %s, target_size: %s',
                    features.shape, targets.shape)
        self._l_r = LinearRegression()
        self._l_r.fit(features, targets)
        logger.info('fit finished.')

    def predict(self, features: np.ndarray) -> np.ndarray:
        '''
        Params
        features: feature vectors
        returns
        dst: predicted vectors
        '''
        logger.info('prediction started.')
        if features.max() > 1 or features.min() < 0:
            logger.warning('features are not scaled. do preprocessing.')
        dst = self._l_r.predict(features)
        logger.info('prediction finished.')
        return dst

    def save_model(self, file_name):
        '''
        save_model(file_name)
        Params
        file_name: str of file path
        '''
        prm_io = pp.ParamIO(file_name)
        prm_io.save(self._l_r)

    def load_model(self, file_name: str):
        '''
        load_model(file_name)
        Params
        file_name: str of file path
        '''
        prm_io = pp.ParamIO(file_name)
        self._l_r = prm_io.load()

    def get_model(self):
        '''
        Returns: model object
        '''
        return self._l_r


if __name__ == '__main__':
    l_r = Model()
    print(l_r.get_model())
