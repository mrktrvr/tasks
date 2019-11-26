'''
dumy_data.py
'''

import numpy as np
import pandas as pd


def dummy_data():
    '''
    dummy pd.DataFrame for test
    '''
    idx = pd.to_datetime(pd.date_range('2010-01-01 00:00:00',
                                       '2011-02-02 23:00:00', freq='H'))
    dummy_df = pd.DataFrame(
        1000 * np.random.rand(len(idx)), index=idx, columns=['dummy'])
    return dummy_df


if __name__ == '__main__':
    df_dummy = dummy_data()
    print(df_dummy.head())
