import os
import sys

from typing import Tuple
from typing import Any

import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


cdir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(cdir, '..'))
from utils.logger import logger
import utils.data_handler as dh


def data_preparation(
        df_src: pd.DataFrame,
        norm_prm_file_name: str,
        n_days: int) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            pd.DatetimeIndex, pd.DatetimeIndex]:
    '''
    Params
    df_src: source dataframe
    norm_prm_file_name: parameter file name for normalisation
    Returns:
    feat_train: feature vectors for train
    tar_train: target vectors for train
    feat_test: feature vectors for test
    tar_test: target vectors for test
    '''
    df_scaled = df_min_max_norm(df_src, norm_prm_file_name, False)
    df_train, df_test = split_df_train_test(df_scaled, n_days)
    df_f1_train, df_f2_train, df_tar_train = split_df_feats_and_tars(df_train)
    df_f1_test, df_f2_test, df_tar_test = split_df_feats_and_tars(df_test)
    feat_train = np.concatenate([df_f1_train.values, df_f2_train.values], 1)
    feat_test = np.concatenate([df_f1_test.values, df_f2_test.values], 1)
    tar_train = df_tar_train.values
    tar_test = df_tar_test.values
    idx_train = df_tar_train.index
    idx_test = df_tar_test.index
    return feat_train, tar_train, feat_test, tar_test, idx_train, idx_test


def split_df_train_test(
        df_src: pd.DataFrame,
        n_days: int,
        dev_rate: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    df_train, df_test = split_df_train_test(df_src)
    split src data frame into 2 parts (train and test)
    Params
    df_src: original data
    Returns
    df_train:
    df_test
    '''
    hours = [i for i, x in enumerate(df_src.index) if x.hour == 0]
    bgn_idx = hours[0]
    feat_len = 24 * n_days
    if bgn_idx != 0:
        logger.info('data shifting to align hour 0.')
    valid_d_size = df_src.shape[0] - bgn_idx + 1
    split_idx = valid_d_size * dev_rate
    split_idx = int(split_idx - (split_idx % feat_len))
    mid_idx = bgn_idx + split_idx
    df_train = df_src.loc[df_src.index[bgn_idx:mid_idx]]
    df_test = df_src.loc[df_src.index[mid_idx:]]
    return df_train, df_test


def array_min_max_norm(arr_src: np.ndarray,
                       prm_file_name: str, do_fit: bool = True) -> np.ndarray:
    '''
    arr_scaled = arr_min_max_norm(arr_src, 'test1.pkl', True)
    arr_scaled = arr_min_max_norm(arr_src, 'test2.pkl', False)

    Params
    arr_src: source np.array
    prm_file_name: parameter file name
    do_fit: parames are obtained by fit if True, else params are from file.
    Returns
    arr_scaled: np.array scaled between 0 and 1
    '''
    min_max_norm = MinMaxNorm(prm_file_name)
    if not os.path.exists(prm_file_name):
        logger.warning('parame file %s does not exist. do fit', prm_file_name)
        do_fit = True
    if do_fit:
        logger.info('fit and transform')
        arr_scaled = min_max_norm.fit_transform(arr_src)
        min_max_norm.save_param()
    else:
        logger.info('transform using parameters from file')
        min_max_norm.load_param()
        arr_scaled = min_max_norm.transform(arr_src)
    return arr_scaled


def array_inversed_min_max_norm(arr_src, prm_file_name):
    min_max_norm = MinMaxNorm(prm_file_name)
    min_max_norm.load_param()
    val_inversed = min_max_norm.inverse_transform(arr_src)
    return val_inversed


def df_min_max_norm(df_src: pd.DataFrame,
                    prm_file_name: str, do_fit: bool = True) -> pd.DataFrame:
    '''
    df_scaled = df_min_max_norm(df_src, 'test1.pkl', True)
    df_scaled = df_min_max_norm(df_src, 'test2.pkl', False)

    Params
    df_src: source DataFrame
    prm_file_name: parameter file name
    do_fit: parames are obtained by fit if True, else params are from file.
    Returns
    df_scaled: DataFrame scaled between 0 and 1
    '''
    arr_src = df_src.values
    arr_scaled = array_min_max_norm(arr_src, prm_file_name, do_fit)
    df_scaled = pd.DataFrame(
        arr_scaled, index=df_src.index, columns=df_src.columns)
    return df_scaled


def df_inversed_min_max_norm(
        df_src: pd.DataFrame, prm_file_name: str) -> pd.DataFrame:
    '''
    df_scaled = df_min_max_norm(df_src, 'test1.pkl')

    Params
    df_src: source DataFrame, scaled between 0 and 1
    prm_file_name: parameter file name
    Returns
    df_scaled: DataFrame with original scale
    '''
    if not os.path.exists(prm_file_name):
        logger.warning('parame file %s does not exist. do fit', prm_file_name)
    logger.info('transform using parameters from file')
    arr_src = df_src.values
    val_inversed = array_inversed_min_max_norm(arr_src, prm_file_name)
    df_scaled = pd.DataFrame(
        val_inversed, index=df_src.index, columns=df_src.columns)
    return df_scaled


def split_df_feats_and_tars(df_src: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    df_feat1, df_feat2, df_targets = split_df_feats_and_tars(df_src)

    Params
    df_src: source DataFrame

    Returns
    df_feat1: power consumptions from n_days before target
    df_feat2: diff of power consumptions from n_days before target
    df_targets: power consumptions of targets
    '''
    n_days = 7
    ib_features = 0
    ie_features = ib_features + n_days * 24
    ib_target = ie_features
    ie_target = ib_target + 24
    feat1 = []
    feat2 = []
    target = []
    dst_index = []
    while ie_target < df_src.shape[0]:
        df_feat = df_src.loc[df_src.index[ib_features:ie_features], :]
        feat1_tmp, feat2_tmp = df_to_feats(df_feat.values)
        feat1.append(feat1_tmp)
        feat2.append(feat2_tmp)
        df_tar = df_src.loc[df_src.index[ib_target:ie_target], :]
        target.append(df_tar.values.flatten())
        dst_index.append(df_tar.index[0])
        ib_features += 24
        ie_features = ib_features + n_days * 24
        ib_target = ie_features
        ie_target = ib_target + 24
    df_feat1 = pd.DataFrame(feat1, index=dst_index)
    df_feat2 = pd.DataFrame(feat2, index=dst_index)
    df_targets = pd.DataFrame(target, index=dst_index)
    df_feat1 = df_feat1.interpolate()
    df_feat2 = df_feat2.interpolate()
    df_targets = df_targets.interpolate()
    return df_feat1, df_feat2, df_targets


def df_to_feats(src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Params
    src: np.array, size of data_length
    Returrns
    feat1: power consumptions, size of data_length
    feat2: diff of power consumptions, size of data_length - 1
    '''
    feat1 = src.flatten()
    feat2 = np.diff(feat1)
    return feat1, feat2


class ParamIO():
    '''
    ParamIO class

    pio = ParamIO(param_file_name)
    pio.save(param)
    param =pio.loat()
    '''

    def __init__(self, param_file_name: str):
        '''
        param_file_name: file anme, string
        '''
        self._param_file_name = param_file_name

    def save(self, param: Any):
        '''
        save param by pickle
        Params
        param: variable to save
        '''
        if param is None:
            logger.error('param is None')
            return
        if self._param_file_name is None:
            logger.warning('file name is None')
            return
        if os.path.exists(self._param_file_name):
            logger.warning('%s exists. overwriting.' % self._param_file_name)
        dir_name = os.path.dirname(self._param_file_name)
        dir_name = '.' if dir_name == '' else dir_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info('%s created.' % dir_name)

        try:
            with open(self._param_file_name, 'wb') as f:
                pickle.dump(param, f)
                logger.info('%s saved.' % self._param_file_name)
        except IOError as exc:
            raise exc

    def load(self) -> Any:
        '''
        load param by pickle
        Returns
        param: variable to save
        '''
        try:
            with open(self._param_file_name, 'rb') as f:
                param = pickle.load(f)
                logger.info('%s loaded.' % self._param_file_name)
        except IOError as exc:
            param = None
            raise exc
        return param


class MinMaxNorm(ParamIO):
    '''
    min-max normalization
    '''

    def __init__(self, param_file_name: str) -> str:
        '''
        Params
        param_file_name: param file name, string

        Attributes:
        _param_file_name: param file path, string
        _param: parameter to transform, dict
        '''
        self._param_file_name = param_file_name
        self._param = None

        super(MinMaxNorm, self).__init__(self._param_file_name)

    def fit_transform(self, src: np.ndarray) -> np.ndarray:
        '''
        min-max normalisation of src with min and max from src

        Params
        src: np.array, size of data length

        Returns
        dst: np.array, size of data length
        '''
        logger.info('MinMax normalisation calculating params')
        src = np.atleast_1d(src)
        try:
            self._param = (min(src), max(src))
        except Exception as exc:
            raise exc
        logger.info('MinMax normalisation calculating params done')
        dst = self.transform(src)
        return dst

    def transform(self, src: np.ndarray) -> np.ndarray:
        '''
        min-max normalisation of src with stored min and max
        Params
        src: np.array, size of data length
        Returns
        dst: np.array, size of data length
        '''
        logger.info('MinMax normalisation transforming the data')
        src = np.atleast_1d(src)
        try:
            dst = (src - self._param[0]) / (self._param[1] - self._param[0])
        except Exception as exc:
            raise exc
        logger.info('MinMax normalisation done')
        return dst

    def inverse_transform(self, src: np.ndarray) -> np.ndarray:
        '''
        inverse min-max normalisation of src with stored min and max
        Params
        src: np.array, size of data length
        Returns
        dst: np.array, size of data length
        '''
        logger.info('Inverse MinMax normalisation transforming the data')
        src = np.atleast_1d(src)
        try:
            dst = src * (self._param[1] - self._param[0]) + self._param[0]
        except Exception as exc:
            raise exc
        logger.info('MinMax normalisation done')
        return dst

    def param(self) -> Tuple[float, float]:
        '''
        parameter interface
        Returns
        dst: tuple of min and max
        '''
        dst = self._param
        return dst

    def set_param(self, param: Tuple[float, float]):
        '''
        set parameters
        param: tuple of min and max
        '''
        self._param = param

    def save_param(self):
        '''
        save param (min_max)
        '''
        self.save(self._param)

    def load_param(self):
        '''
        load param (min_max)
        '''
        self._param = self.load()



def post_process_array_to_df(feat, pred, tar, n_days, idx):
    n_days = pred.shape[1]
    df_feat_norm = pd.DataFrame(feat[:, :n_days * n_days], index=idx)
    df_pred_norm = pd.DataFrame(pred, index=idx)
    df_tar_norm = pd.DataFrame(tar, index=idx)
    return df_feat_norm, df_pred_norm, df_tar_norm


def post_process_inverse_norm(
        df_feat_norm, df_pred_norm, df_tar_norm, norm_prm_file_name):
    df_feat = df_inversed_min_max_norm(df_feat_norm, norm_prm_file_name)
    df_pred = df_inversed_min_max_norm(df_pred_norm, norm_prm_file_name)
    df_tar = df_inversed_min_max_norm(df_tar_norm, norm_prm_file_name)
    return df_feat, df_pred, df_tar


def post_process_array_to_df(feat, pred, tar, n_days, idx):
    df_feat_norm = pd.DataFrame(feat[:, :24 * n_days], index=idx)
    df_pred_norm = pd.DataFrame(pred, index=idx)
    df_tar_norm = pd.DataFrame(tar, index=idx)
    return df_feat_norm, df_pred_norm, df_tar_norm


def post_process_inverse_norm(df_feat_norm, df_pred_norm, df_tar_norm, norm_prm_file_name):
    df_feat = df_inversed_min_max_norm(df_feat_norm, norm_prm_file_name)
    df_pred = df_inversed_min_max_norm(df_pred_norm, norm_prm_file_name)
    df_tar = df_inversed_min_max_norm(df_tar_norm, norm_prm_file_name)
    return df_feat, df_pred, df_tar
