import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..', 'hec'))
import utils.data_handler as dh
from utils.eval import match_rate


def plot_all_seq_all_data(data):
    xmin = min([x.index.min() for x in data.values()])
    xmax = max([x.index.max() for x in data.values()])
    for data_name in data.keys():
        df = data[data_name]
        print('dataset:%s, full sequence of power consumptions' % data_name)
        plot_all_seq(df, data_name, xmin, xmax)
        plt.show()


def plot_day_of_week(df_src, tar_month, data_name):
    df_wo_hds = dh.get_df_not_holidays(df_src)
    dows = range(7)
    plot_src = {}
    for i, tar_dow in enumerate(dows, 1):
        df_dow_month = dh.get_df_dow_in_month(df_wo_hds, tar_dow, tar_month)
        plot_src.update({tar_dow: dh.reshape_day_by_day(df_dow_month)})
    ymin = np.nanmin([x.values.min() for x in plot_src.values()])
    ymax = np.nanmax([x.values.max() for x in plot_src.values()])
    n_rows = 2
    n_cols = len(plot_src)
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 4))
    for i, (dow, val) in enumerate(sorted(plot_src.items()), 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        plot_daily_all(ax, val, ymin, ymax, i, dow, data_name)
        ax = fig.add_subplot(n_rows, n_cols, i + n_cols)
        plot_daily_ave_med_std(ax, val, ymin, ymax, dow, data_name)


def plot_seasonaly(data, tar_months):
    for data_name, data_df in data.items():
        plot_src = {}
        print('dataset:%s, daily consumptions in %s' %
              (data_name,
               ', '.join([dh.month_no_to_name(m) for m in tar_months])))
        for i, m in enumerate(tar_months, 1):
            df_month = dh.get_df_month(data_df, m)
            plot_src.update({m: dh.reshape_day_by_day(df_month)})
        ymin = np.nanmin([x.values.min() for x in plot_src.values()])
        ymax = np.nanmax([x.values.max() for x in plot_src.values()])
        n_rows = 2
        n_cols = len(tar_months)
        fig = plt.figure(figsize=(len(tar_months) * 6, n_rows * 4))
        for i, (m, v) in enumerate(sorted(plot_src.items()), 1):
            ax = fig.add_subplot(n_rows, len(tar_months), i)
            plot_daily_all(ax, v, ymin, ymax, i, m, data_name)
            ax = fig.add_subplot(n_rows, len(tar_months), i + len(tar_months))
            plot_daily_ave_med_std(ax, v, ymin, ymax, m, data_name)
            # ax = fig.add_subplot(n_rows, n_cols, i + 2 * len(tar_months))
            # plot_daily_ave(ax, v, ymin, ymax, m, data_name)
        plt.show()


def plot_daily_ave(ax, v, ymin, ymax, m, data_name):
    ax.plot(v.values.mean(1), 'x')
    ax.grid(True)
    ax.set_ylim(ymin, ymax)
    ax.set_title('%s daily average in %2d' % (data_name, m))
    ax.set_ylabel('powers')
    ax.set_xlabel('days')
    plt.tight_layout()


def plot_daily_all(ax, v, ymin, ymax, i, m, data_name):
    ax.plot(v.T)
    ax.grid(True)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-1, 24)
    ax.set_xlabel('hours')
    ax.set_ylabel('powers')
    ax.set_title('powers in days of %2d' % m)
    if i == 1:
        ax.text(-5, ymax, data_name, fontsize='xx-large', va='bottom')


def plot_daily_ave_med_std(ax, v, ymin, ymax, m, data_name):
    v_mean = v.mean(0)
    v_std = v.std(0)
    v_med = v.median(0)
    x = range(v.shape[-1])
    ax.fill_between(
        x, v_mean - v_std, v_mean + v_std, alpha=0.5, label='1-$\sigma$')
    v_med.plot(legend=False, ax=ax, label='median')
    v_mean.plot(legend=False, ax=ax, label='average')
    ax.grid(True)
    ax.set_xlim(-1, 24)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('hours')
    ax.set_ylabel('powers')
    ax.set_title('ave/med/std %s of %2d' % (data_name, m))
    ax.legend(loc=0)


def plot_all_data(data):
    xmin = min([x.index.min() for x in data.values()])
    xmax = max([x.index.max() for x in data.values()])
    for i, (k, df) in enumerate(data.iteritems()):
        plot_all_seq(df, k, xmin, xmax)


def plot_all_seq(df, data_name, xmin=None, xmax=None):
    fig = plt.figure(figsize=(16, 4))
    years = pd.unique(df.index.year)
    ybgn = years[0]
    ax = fig.add_subplot(1, 1, 1)
    for yend in years[1:]:
        d = df[(str(ybgn) <= df.index) & (df.index < str(yend))]
        d.columns = ['%s_%s-%s' % (d.columns[0], ybgn, yend)]
        d.plot(ax=ax)
        ybgn = yend
    ax.legend(loc=0, ncol=int(len(years) / 3))
    ax.grid(True)
    xmin = df.index.min() if xmin is None else xmin
    xmax = df.index.max() if xmax is None else xmax
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin=0)
    ax.set_title(data_name)


def plot_regression_res(X_test, Y_test, estim, n_day_feat, n_plots=10):
    n_col = 2
    n_row = int(np.ceil(float(n_plots) / n_col))
    fig = plt.figure(figsize=(8 * n_col, 4 * n_row))
    for i, (idx, est_val) in enumerate(estim.iterrows(), 1):
        x_i = X_test.iloc[i]
        y_i = Y_test.iloc[i]
        ax = fig.add_subplot(n_row, n_col, i)
        for j, xx in enumerate(x_i.values.reshape(n_day_feat, 24)):
            date = x_i.name - pd.to_timedelta(n_day_feat - j, unit='d')
            date = date.strftime('%Y-%m-%d')
            ax.plot(xx, '-', label='input %s' % date, lw=0.3)
        ax.plot(
            y_i.values, 'x-', label='truth %s' % y_i.name, lw=2, color='red')
        ax.plot(est_val, '+-', label='estim', lw=2, color='blue')
        ax.set_title(
            'from %s to %s to estimate %s' %
            (x_i.name.date() - pd.Timedelta('%d days' % n_day_feat),
             x_i.name.date() - pd.Timedelta('%d days' % 1),
             y_i.name.strftime('%Y-%m-%d(%a)')))
        ax.grid(True)
        rmse = np.sqrt(mean_squared_error(y_i.values, est_val))
        mr = match_rate(y_i.values, est_val)
        xmin, _, _, ymax = ax.axis()
        txt_str = '%s: %.3f\n%s: %.3f' % ('RMSE', rmse, 'MR', mr)
        ax.text(xmin, ymax, txt_str, va='top', ha='left')
        ax.legend(loc=0, fontsize='x-small', ncol=2)
        if i >= n_plots:
            break


def plot_rmse_day_by_day(Y_test, estim):
    rmse_seq = [np.sqrt(np.mean((estim.values[i] - Y_test.values[i])**2))
                for i in range(Y_test.shape[0])]
    rmse = pd.DataFrame(rmse_seq, index=Y_test.index)
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(1, 1, 1)
    rmse.plot(ax=ax, legend=None)
    rmse_all = np.sqrt(mean_squared_error(Y_test.values, estim.values))
    ax.set_title('RMSE day bay day RMSE all: %.3f' % rmse_all)
    ax.set_xticks(rmse.index[rmse.index.weekday == 0])
    xticklbls = [
        x.strftime('%Y-%m-%d(%a)') for x in rmse.index[rmse.index.weekday == 0]
    ]
    ax.set_xlim(rmse.index[0], rmse.index[365])
    ax.set_xticklabels(xticklbls, rotation=90)
    ax.grid()


def plot_stats_data_comparison(data):
    min_list = [data[k].min().values[0] for k in data.keys()]
    max_list = [data[k].max().values[0] for k in data.keys()]
    mean_list = [data[k].mean().values for k in data.keys()]
    std_list = [data[k].std().values[0] for k in data.keys()]
    med_list = [data[k].median().values[0] for k in data.keys()]
    n_cols = 2
    n_rows = 3
    fig = plt.figure(1, figsize=(6 * n_cols, 4 * n_rows))
    ax = fig.add_subplot(n_rows, n_cols, 1)
    plot_stats(min_list, max_list, mean_list, std_list, med_list, ax)
    _ = ax.set_xticks(range(len(data)))
    _ = ax.set_xticklabels(data.keys(), rotation=90)
    _ = ax.set_ylim(0, 65000)
    _ = ax.set_title('min, max, mean and std of all months')

    for i, tar_month in enumerate([1, 4, 7, 10], 3):
        min_list = []
        max_list = []
        mean_list = []
        std_list = []
        med_list = []
        for k in data.keys():
            dfm = dh.get_df_month(data[k], tar_month)
            min_list.append(dfm.min().values[0])
            max_list.append(dfm.max().values[0])
            mean_list.append(dfm.mean().values[0])
            std_list.append(dfm.std().values[0])
            med_list.append(dfm.median().values[0])
        ax = fig.add_subplot(n_rows, n_cols, i)
        plot_stats(min_list, max_list, mean_list, std_list, med_list, ax)
        month_name = dh.month_no_to_name(tar_month)
        _ = ax.set_title('month:%s' % month_name)
        _ = ax.set_xticks(range(len(data)))
        _ = ax.set_xticklabels(data.keys(), rotation=90)
        _ = ax.set_ylim(0, 65000)
    plt.tight_layout()


def plot_stats(mins, maxs, means, stds, meds, ax=None, linestyle=''):
    '''
    plot_stats(mins, maxs, means, stds, meds, ax=None, linestyle='')
    '''
    if ax is None:
        fig_idx = 1
        n_rows = 1
        n_cols = 1
        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
        ax = fig.add_subplot(n_rows, n_cols, fig_idx)
    n_data = len(mins)
    xpoints = np.arange(n_data)
    _ = ax.plot(
        xpoints, maxs,
        linestyle=linestyle, marker='^', color='red', label='max')
    _ = ax.plot(
        xpoints, meds,
        linestyle=linestyle, marker='*', color='pink', label='median')
    _ = ax.errorbar(
        xpoints, means,
        yerr=stds, linestyle=linestyle, marker='x', label='mean and std')
    _ = ax.plot(
        xpoints, mins,
        linestyle=linestyle, marker='v', color='blue', label='min')
    _ = ax.legend(loc=0)
    _ = ax.grid(True)


def plot_day_of_week_all(data, tar_months):
    for tar_month in tar_months:
        print('Month: %s' % dh.month_no_to_name(tar_month))
        for data_name in data.keys():
            df = data[data_name]
            print('dataset:%s, Monday to Sunday in %s' %
                  (data_name, dh.month_no_to_name(tar_month)))
            plot_day_of_week(df, tar_month, data_name)
            plt.show()


def plot_stats_monthly(data):
    n_data = len(data)
    n_cols = 2
    n_rows = int(np.ceil(n_data / float(n_cols)))
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i, k in enumerate(data.keys(), 1):
        mins = []
        maxs = []
        means = []
        stds = []
        meds = []
        for tar_month in range(1, 13):
            df_month = dh.get_df_month(data[k], tar_month)
            mins.append(df_month.min().values[0])
            maxs.append(df_month.max().values[0])
            means.append(df_month.mean().values[0])
            stds.append(df_month.std().values[0])
            meds.append(df_month.median().values[0])
        ax = fig.add_subplot(n_rows, n_cols, i)
        plot_stats(mins, maxs, means, stds, meds, ax, '-')
        _ = ax.set_title('dataset:%s' % k)
        _ = ax.set_xticks(range(12))
        _ = ax.set_xticklabels(range(1, 13), rotation=90)
        _ = ax.set_ylim(ymin=0)
    plt.show()
    plt.tight_layout()


def plot_stats_weekly(data):
    n_data = len(data)
    weekday_names = [dh.week_no_to_name(i) for i in range(7)]
    n_cols = 2
    n_rows = int(np.ceil(n_data / float(n_cols)))
    for tar_month in [1, 5, 7, 10]:
        print('month %d' % tar_month)
        fig = plt.figure(tar_month, figsize=(6 * n_cols, 4 * n_rows))
        for i, (data_name, data_df) in enumerate(data.items(), 1):
            df_month = dh.get_df_month(data_df, tar_month)
            mins = []
            maxs = []
            means = []
            stds = []
            meds = []
            for tar_day in range(7):
                df_weekday = dh.get_df_dayofweek(df_month, tar_day)
                mins.append(df_weekday.min().values[0])
                maxs.append(df_weekday.max().values[0])
                means.append(df_weekday.mean().values[0])
                stds.append(df_weekday.std().values[0])
                meds.append(df_weekday.median().values[0])
            ax = fig.add_subplot(n_rows, n_cols, i)
            plot_stats(mins, maxs, means, stds, meds, ax, '-')
            month_name = dh.month_no_to_name(tar_month)
            _ = ax.set_title('dataset:%s, month:%s' % (data_name, month_name))
            _ = ax.set_xticks(range(7))
            _ = ax.set_xticklabels(weekday_names, rotation=90)
            _ = ax.set_ylim(ymin=0)
        plt.tight_layout()
        plt.show()


def plot_diff_days_hours_all(data):
    for data_name, data_df in data.items():
        plot_diff_days_hours(data_name, data_df)


def plot_diff_days_hours_each_month(data, tar_months):
    for data_name, data_df in data.items():
        for month_no in tar_months:
            df_month = dh.get_df_month(data_df, month_no)
            dname_month = '%s(%s)' % (data_name, dh.month_no_to_name(month_no))
            plot_diff_days_hours(dname_month, df_month)


def plot_diff_days_hours(data_name, data_df):
    print(data_name)
    df_d = dh.reshape_day_by_day(data_df)
    fig = plt.figure(figsize=(12, 4))
    # --- diff between days
    ax = fig.add_subplot(1, 2, 1)
    df_d_diff = df_d.diff(axis=0)
    mins = df_d_diff.min()
    maxs = df_d_diff.max()
    aves = df_d_diff.mean()
    stds = df_d_diff.std()
    meds = df_d_diff.median()
    _ = ax.set_title('stats of difference between days at same hour')
    plot_stats(mins, maxs, aves, stds, meds, ax=ax)
    # --- diff between hours
    _ = ax.set_ylim(-2000, 2000)
    ax = fig.add_subplot(1, 2, 2)
    df_d_diff = df_d.diff(axis=1)
    mins = df_d_diff.min()
    maxs = df_d_diff.max()
    aves = df_d_diff.mean()
    stds = df_d_diff.std()
    meds = df_d_diff.median()
    plot_stats(mins, maxs, aves, stds, meds, ax=ax)
    _ = ax.set_title('stats of difference between hours')
    _ = ax.set_ylim(-2000, 2000)
    plt.show()


def plot_features_and_targets(features, targets, n_samples):
    for i in range(n_samples):
        print('sample no %d' % i)
        fig = plt.figure(figsize=(18, 3))
        ax = plt.subplot2grid((1, 3), (0, 0))
        _ = ax.plot(features[i, :168])
        _ = ax.grid(True)
        _ = ax.set_title('power consumptions during 7 days before target days')
        ax = plt.subplot2grid((1, 3), (0, 1))
        _ = ax.plot(features[i, 168:])
        _ = ax.grid(True)
        _ = ax.set_title('Difference between hours in 7 days')
        ax = plt.subplot2grid((1, 3), (0, 2))
        _ = ax.plot(targets[i])
        _ = ax.grid(True)
        _ = ax.set_title('Target power consumption')
        plt.show()
