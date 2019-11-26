'''
predict_api.py
'''

import os
import sys

import argparse

from flask import Flask
from flask import request
from flask import jsonify

cdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cdir, '..'))

from utils.logger import logger
import model.preprocessing as pp
from model.model import Model

logger.setLevel('DEBUG')


if __name__ == '__main__':
    mdl_prm_name = sys.argv[1]
    norm_prm_name = sys.argv[2]

    app = Flask(__name__)

    lr_model = Model()
    lr_model.load_model(mdl_prm_name)

    @app.route('/')
    def root():
        '''
        root
        '''
        return 'root'

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        '''
        pethod to predict duration
        URL/predictt?dta=#####
        '''
        # --- parameter
        logger.debug('getting paramters')
        data = request.args.get('data')
        logger.debug('data size: %s', data.shape)

        # --- preprocessin
        arr_src = pp.array_min_max_norm(data, norm_prm_name)

        # --- prediction
        logger.debug('prediction')
        prediction = lr_model.predict(arr_src)

        # --- post processing
        arr_dst = pp.array_min_max_norm(prediction, norm_prm_name)

        # --- result
        logger.debug('prediction')
        dst = jsonify({'prediction': arr_dst.tolist()})
        logger.debug(dst)
        return dst

    app.run()
