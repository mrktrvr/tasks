'''
unit_test_data_handler.py
'''
import unittest

import os
import sys
import numpy as np

cdir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(cdir, '..', 'hec')
sys.path.append(lib_dir)

import utils.data_handler as dh
from utils.logger import logger
logger.setLevel('CRITICAL')


class TestDataHandler(unittest.TestCase):
    '''
    Unit test of data handler modules
    '''
    def test_data_loader(self):
        data_dir = os.path.join(cdir, '..', 'data')

        with self.assertRaises(Exception):
            dst_df = dh.load_csv_to_df('file_name.csv')

        file_path = os.path.join(data_dir, 'AEP_hourly.csv')
        if not os.path.exists(file_path):
            logger.critical('%s does not exist. test skipped.', file_path)
            dst_df = dh.load_csv_to_df(file_path)
            self.assertEqual(dst_df.shape, (121296, 1))

        if not os.path.exists(data_dir):
            logger.critical('%s does not exist. test skipped.', data_dir)
        else:
            data = dh.data_loader(data_dir, 12)
            self.assertEqual(len(data), 11)
            self.assertEqual(data['AEP_hourly'].shape, (121296, 1))
            self.assertEqual(data['COMED_hourly'].shape, (66504, 1))
            self.assertEqual(data['DAYTON_hourly'].shape, (121296, 1))
            self.assertEqual(data['DEOK_hourly'].shape, (57744, 1))
            self.assertEqual(data['DOM_hourly'].shape, (116208, 1))
            self.assertEqual(data['DUQ_hourly'].shape, (119088, 1))
            self.assertEqual(data['EKPC_hourly'].shape, (45336, 1))
            self.assertEqual(data['FE_hourly'].shape, (62880, 1))
            self.assertEqual(data['NI_hourly'].shape, (58464, 1))
            self.assertEqual(data['PJME_hourly'].shape, (145392, 1))
            self.assertEqual(data['PJMW_hourly'].shape, (143232, 1))

    def test_no_to_str(self):
        self.assertEqual(dh.week_no_to_name(-1), 'Unknown')
        self.assertEqual(dh.week_no_to_name(0), 'Mon')
        self.assertEqual(dh.week_no_to_name(1), 'Tue')
        self.assertEqual(dh.week_no_to_name(2), 'Wed')
        self.assertEqual(dh.week_no_to_name(3), 'Thu')
        self.assertEqual(dh.week_no_to_name(4), 'Fri')
        self.assertEqual(dh.week_no_to_name(5), 'Sat')
        self.assertEqual(dh.week_no_to_name(6), 'Sun')
        self.assertEqual(dh.week_no_to_name(7), 'Unknown')
        self.assertEqual(dh.month_no_to_name(-1), 'Unknown')
        self.assertEqual(dh.month_no_to_name(1), 'January')
        self.assertEqual(dh.month_no_to_name(4), 'April')
        self.assertEqual(dh.month_no_to_name(7), 'July')
        self.assertEqual(dh.month_no_to_name(10), 'October')
        self.assertEqual(dh.month_no_to_name(13), 'Unknown')

    def test_reindex(self):
        df_src = dummy_data()
        drop_idx = df_src.index[3:5]
        df_dst = dh.df_reindex(df_src.drop(index=drop_idx))
        self.assertEqual(df_dst.shape, df_src.shape)

    def test_filters(self):
        # --- filter quantaile
        df_src = dummy_data()
        df_new = dh.filter_quantaile(df_src)
        self.assertTrue(df_src.shape[0] == df_new.shape[0])
        self.assertFalse(df_src.max() in df_new[df_new.columns[0]])
        self.assertAlmostEqual(df_src.min().values[0],
                               df_new[df_new.columns[0]].min())
        # --- df holidays
        df_src = dummy_data()
        df_new = dh.get_df_holidays(df_src)
        self.assertEqual(df_new.shape[0] / 24, 12)

        # --- df not holidays
        df_src = dummy_data()
        df_new = dh.get_df_not_holidays(df_src)
        self.assertEqual(df_new.shape[0] / 24, 386)
        self.assertEqual(df_new.shape[1], 1)

        # --- df target month
        df_src = dummy_data()
        df_new = dh.get_df_month(df_src, 1)
        self.assertEqual(df_new.shape[0] / 24, 62)
        self.assertEqual(df_new.shape[1], 1)
        df_new = dh.get_df_month(df_src, 2)
        self.assertEqual(df_new.shape[0] / 24, 30)
        self.assertEqual(df_new.shape[1], 1)
        df_new = dh.get_df_month(df_src, 13)
        self.assertEqual(df_new.shape[0], 0)
        self.assertEqual(df_new.shape[1], 1)

        # --- df target dayofweek
        df_src = dummy_data()
        df_new = dh.get_df_dayofweek(df_src, 1)
        self.assertEqual(df_new.shape[0], 1368)
        self.assertEqual(df_new.shape[1], 1)
        df_new = dh.get_df_dayofweek(df_src, 2)
        self.assertEqual(df_new.shape[0], 1368)
        self.assertEqual(df_new.shape[1], 1)
        df_new = dh.get_df_dayofweek(df_src, 7)
        self.assertEqual(df_new.shape[0], 0)
        self.assertEqual(df_new.shape[1], 1)

        # --- df df weekdays
        df_src = dummy_data()
        df_new = dh.get_df_weekdays(df_src)
        self.assertNotEqual(df_new.shape, df_src.shape)
        self.assertTrue(all([x in range(0, 5) for x in df_new.index.dayofweek]))
        self.assertFalse(any([x in [5, 6] for x in df_new.index.dayofweek]))

    def test_reshapes(self):
        df_src = dummy_data()
        df_d = dh.reshape_day_by_day(df_src)
        self.assertEqual(df_d.shape, (396, 24))
        self.assertTrue(np.allclose(df_d.columns, np.arange(0, 24)))

        df_src = dummy_data()
        df_w = dh.reshape_week_by_week(df_src)
        self.assertEqual(df_w.shape, (56, 168))
        self.assertTrue(np.allclose(df_w.columns, np.arange(0, 24 * 7)))


if __name__ == '__main__':
    from dummy_data import dummy_data
    unittest.main()
else:
    from .dummy_data import dummy_data
