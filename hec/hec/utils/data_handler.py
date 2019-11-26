import os
import sys
import datetime
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

from .logger import logger

cdir = os.path.abspath(os.path.dirname(__file__))

# cdir = os.getcwd()


def data_loader(data_dir, n_data):
    '''
    data = data_loader()
    '''
    data = {}
    for fname in sorted(os.listdir(data_dir)):
        body, ext = os.path.splitext(fname)
        fpath = os.path.join(data_dir, fname)
        logger.info(fpath)
        if ext == '.csv':
           df_src = load_csv_to_df(fpath)
        else:
            continue
        if df_src is not None:
            data.update({body: df_src.sort_index()})
        if len(data) >= n_data:
            break
    # --- print data info
    logger.info('total: %d dataset' % len(data))
    logger.info('no| dataset names')
    logger.info('%2s|%20s' % ('--', '-' * 20))
    for i, (data_name, d_f) in enumerate(data.items(), 1):
            logger.info('%2d| %s %s' % (i, data_name, d_f.shape))
    return data


def load_csv_to_df(file_name, data_size_th=None):
    '''
    df = load_csv_to_df(file_name)
    '''
    data_size_th = 24 * 365 * 4 if data_size_th is None else data_size_th

    df = pd.read_csv(file_name, index_col=0)
    logger.info('%s loaded' % file_name)
    # --- index using datetime
    df.index = pd.to_datetime(df.index)
    if df.shape[-1] != 1:
        logger.warning('%s contains some columns. skipped.' % file_name)
        return None
    if df.shape[0] < data_size_th:
        logger.warning('%s is shorter than %d. skipped.' %
                       (file_name, data_size_th))
        return None
    # --- index for empty values (summer time problem)
    df_new = df_reindex(df)
    return df_new

def df_reindex(df_src: pd.DataFrame) -> pd.DataFrame:
    '''
    df_dst = df_reindex(df_src)
    Param
    df_src: pd.DataFrame
    df_dst: pd.DataFrame
    '''
    new_idx = pd.to_datetime(
        pd.date_range(df_src.index.min(), df_src.index.max(), freq='H'))
    df_dst = pd.DataFrame(index=new_idx, columns=df_src.columns)
    df_dst.loc[df_src.index, :] = df_src.values
    return df_dst

def filter_quantaile(df: pd.DataFrame = None, q: float = 0.95) -> pd.DataFrame:
    '''
    df = filter_quantaile(df, q=0.95)
    Params
    df: DataFrame
    Returns
    df: DataFrame
    '''
    quant_val_high = df[df.columns[0]].quantile(q=q)
    df.loc[df[df.columns[0]] > quant_val_high] = quant_val_high
    return df


def get_us_holidays(df: pd.DataFrame = None) -> pd.DataFrame:
    '''
    holidays = get_us_holidays(df)

    '''
    ushc = USFederalHolidayCalendar()

    if df is not None:
        t_min = df.index.min()
        t_max = df.index.max()
        vals = [x for x in ushc.holidays() if (t_min <= x) & (x <= t_max)]
        holidays = pd.to_datetime(vals)
    else:
        holidays = pd.to_datetime(ushc.holidays())

    return holidays


def get_df_holidays(df_src: pd.DataFrame) -> pd.DataFrame:
    '''
    Params
    df_src: source DataFrame
    Returns:
    df_dst: DataFrame which is extracted holidays from df_src
    '''
    holidays = get_us_holidays(df_src)
    idx = [x in holidays for x in df_src.index.date]
    df_dst = df_src.loc[idx]
    return df_dst


def get_df_not_holidays(df_src: pd.DataFrame) -> pd.DataFrame:
    '''
    Params
    df_src: source DataFrame
    Returns:
    df_dst: DataFrame which is extracted holidays from df_src
    '''
    holidays = get_us_holidays(df_src)
    idx = [x not in holidays for x in df_src.index.date]
    df_dst = df_src.loc[idx]
    return df_dst


def get_df_weekdays(df_src: pd.DataFrame) -> pd.DataFrame:
    '''
    Params
    df_src: source DataFrame
    Returns:
    df_dst: DataFrame which is extracted holidays from df_src
    '''
    idx = [x in range(0, 5) for x in df_src.index.dayofweek]
    df_dst = df_src.loc[idx]
    return df_dst


def get_df_month(df_src: pd.DataFrame, tar_month: int) -> pd.DataFrame:
    '''
    Params
    df_src: source DataFrame
    tar_month: 1 to 12
    Returns
    df_dst: DataFrame filtered by tar_month
    '''
    idx = df_src.index.month == tar_month
    df_dst = df_src.loc[idx]
    return df_dst


def get_df_dayofweek(df_src: pd.DataFrame, tar_dayofweek: int) -> pd.DataFrame:
    '''
    filter target day of week.
    df_dst = get_df_dayofweek(df_src, tar_dayofweek)
    Params
    df_src: source DataFrame
    tar_dayofweek: 0 to 6, int, 0 - Mon, 6 - Sun
    Returns:
    df_dst DataFrame filtered by tar_dayofweek
    '''
    idx = df_src.index.dayofweek == tar_dayofweek
    dst = df_src.loc[idx]
    return dst


def get_df_dow_in_month(
        df_src: pd.DataFrame, tar_dayofweek: int, tar_month: int) -> pd.DataFrame:
    '''
    dst_df = get_df_dow_in_month(df_src, tar_dayofweek, tar_month)
    Params
    df_src: source DataFrame
    tar_dayofweek: 0 to 6, int, 0 - Mon, 6 - Sun
    tar_month: 1 to 12
    Returns
    dst_df: DataFrame filtered tar_dayofweek and tar_month
    '''
    tf_dow = df_src.index.dayofweek == tar_dayofweek
    tf_month = df_src.index.month == tar_month
    idx = tf_dow & tf_month
    df_dst = df_src.loc[idx]
    return df_dst


def reshape_day_by_day(df_src):
    '''
    df_reshaped = reshape_day_bay_day(df_src)

    Params
    df_src: DataFrame, index=datetime(hourly), columns=data

    Returns
    df_dst: DataFrame, index=datetime(daily), columns=range(0, 23)
    '''
    midnight_idx_list = sorted(df_src.index[df_src.index.hour == 0])
    ib = midnight_idx_list[0]
    ie = midnight_idx_list[-2]
    tmp = df_src[(ib <= df_src.index) & (df_src.index < ie)]
    v = tmp.values.reshape(int(tmp.shape[0] / 24), 24)
    idx = tmp.index[tmp.index.hour == 0]
    clm = np.arange(0, 24)
    dst = pd.DataFrame(v, index=idx, columns=clm)
    return dst


def reshape_week_by_week(df_src):
    hour_dow_zero = np.all([df_src.index.hour == 0,
                            df_src.index.dayofweek == 0], 0)
    midnight_idx_list = sorted(df_src.index[hour_dow_zero])
    ib = midnight_idx_list[0]
    ie = midnight_idx_list[-1]
    tmp = df_src[(ib <= df_src.index) & (df_src.index < ie)]
    v = tmp.values.reshape((int(tmp.shape[0] / (24 * 7)), 24 * 7))
    idx = tmp.index[::(24 * 7)]
    clm = np.arange(0, 24 * 7)
    dst = pd.DataFrame(v, index=idx, columns=clm)
    return dst


def week_no_to_name(week_no: int) -> str:
    '''
    Params
    week_no: int, 0 to 6, int, 0 - Mon, 6 - Sun
    Returns
    week_name: str, Mon, Tue, ..., Sun
    '''
    if week_no < 0 or week_no > 6:
        logger.warning('unknown week no %d. return Unknown' % week_no)
        week_name = 'Unknown'
    else:
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        week_name = weekday_names[week_no]
    return week_name


def month_no_to_name(month_no: int) -> str:
    '''
    Params
    month_no: int, 0 to 6
    Returns
    month_name: str, January, ..., December
    '''
    if month_no < 1 or month_no > 12:
        logger.warning('unknown month no %d. return Unknown' % month_no)
        month_name = 'Unknown'
    else:
        month_name = datetime.date(1900, month_no, 1).strftime('%B')
    return month_name


def reshape_df_for_regression(df, n_day_feat=7):
    '''
    '''
    prev_year = df.index.max().year - 1
    train_df = df[df.index.year <= prev_year].interpolate()
    test_df = df[prev_year < df.index.year].interpolate()
    df_day_train = reshape_day_by_day(train_df)
    df_day_test = reshape_day_by_day(test_df)
    X_train, Y_train = _reshape_df_core(df_day_train, n_day_feat)
    X_test, Y_test = _reshape_df_core(df_day_test, n_day_feat)
    print(
        'X_train: %s (%s, %s)' %
        (list(X_train.shape), X_train.index[0], X_train.index[-1]))
    print(
        'Y_train: %s (%s, %s)' %
        (list(Y_train.shape), Y_train.index[0], Y_train.index[-1]))
    print(
        'X_test : %s (%s, %s)' %
        (list(X_test.shape), X_test.index[0], X_test.index[-1]))
    print(
        'Y_test : %s (%s, %s)' %
        (list(Y_test.shape), Y_test.index[0], Y_test.index[-1]))
    return X_train, Y_train, X_test, Y_test


def _reshape_df_core(df_day, n_day_feat):
    tmp = [df_day.iloc[i:-(n_day_feat - i)] for i in range(n_day_feat)]
    v = np.concatenate([x.values for x in tmp], 1)
    idx = [x.index[-1] for x in tmp]
    idx = df_day.index[n_day_feat - 1:-1]
    X = pd.DataFrame(v, index=idx)
    Y = df_day.iloc[n_day_feat:]
    return X, Y
