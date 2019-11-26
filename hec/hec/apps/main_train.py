'''
main_train.py
'''

import os
import sys

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))

import model.preprocessing as pp
from utils.logger import logger
import utils.data_handler as dh
import model.model as mdl


def train(csv_file_name, mdl_prm_name, norm_prm_name):
    # --- data loading
    df_src = dh.load_csv_to_df(csv_file_name)
    n_days = 7

    # --- preprocessing
    ret = pp.data_preparation(df_src, norm_prm_name, n_days)
    feat_train, tar_train, feat_test, tar_test, idx_train, idx_test = ret

    # --- training model
    l_r = mdl.Model()
    l_r.fit(feat_train, tar_train)

    # --- store model
    l_r.save_model(mdl_prm_name)


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
        train(csv_file_name, mdl_prm_name, norm_prm_name)


if __name__ == '__main__':
    main()
