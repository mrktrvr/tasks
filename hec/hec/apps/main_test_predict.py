'''
main_predict.py
'''

import os
import sys
import pandas as pd

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))

import model.preprocessing as pp
from utils.logger import logger
import utils.data_handler as dh
import model.model as mdl


def test_predict(csv_file_name, mdl_prm_name, norm_prm_name):
    # --- data loading
    df_src = dh.load_csv_to_df(csv_file_name)
    n_days = 7

    # --- preprocessing
    ret = pp.data_preparation(df_src, norm_prm_name, n_days)
    feat_train, tar_train, feat_test, tar_test, idx_train, idx_test = ret

    # --- loading model
    l_r = mdl.Model()
    l_r.load_model(mdl_prm_name)

    # --- prediction
    estim = l_r.predict(feat_test)

    # --- post process
    df_pred_norm = pd.DataFrame(estim, index=idx_test)
    df_pred = pp.df_inversed_min_max_norm(df_pred_norm, norm_prm_name)
    print(df_pred.shape)


def main():
    '''
    main function
    '''
    if len(sys.argv) != 4:
        logger.error('arguments required')
        print('%s csv_file_name mdl_prm_name norm_prm_name' % sys.argv[0])
    else:
        csv_file_name = sys.argv[1]
        mdl_prm_name = sys.argv[2]
        norm_prm_name = sys.argv[3]
        test_predict(csv_file_name, mdl_prm_name, norm_prm_name)


if __name__ == '__main__':
    main()
