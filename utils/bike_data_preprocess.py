import numpy as np
import pandas as pd
import os
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tsai.data.preparation import SlidingWindowPanel

from algorithms.utils import catch22, extract_manual_features


def season_cols():
    return ["mnth", "hr", "weekday", "holiday", "weathersit"]


def load(data_path="../Data/bike/"):
    """
    """

    data = pd.read_csv(os.path.join(data_path, "hour.csv"))

    def parse_dates(inputs):
        date_str, hour = inputs
        year, month, day = [int(n) for n in date_str.split("-")]
        return datetime.datetime(year=year, month=month, day=day, hour=hour)

    data['timestamp'] = data[["dteday", 'hr']].apply(parse_dates, axis=1)

    # hacky way of creating an id column to distinguish groups of consecutive rows with 1h time diff
    data['timediff'] = data['timestamp'].diff()
    data.at[0, 'timediff'] = data['timediff'][1]
    # mark the 1h timediff rows as zero seconds (-3600), and rows with larger timediff will be positive
    data['id'] = data['timediff'].apply(lambda x: x.seconds - 3600)
    # sum the consecutive pairs of rows, so the id value will change at each step where the timediff is
    # greater than zero
    data['id'] = data['id'].rolling(2, min_periods=1).sum()

    # standardize season cols since they should be known in advance
    data[season_cols()] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data[season_cols()])

    columns_keep = ["id", "timestamp", "season", "mnth", "hr", "weekday", "holiday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"]
    data = data[columns_keep]
    data11 = data[data['timestamp'] < datetime.datetime(year=2012, month=1, day=1)]
    data12 = data[data['timestamp'] >= datetime.datetime(year=2012, month=1, day=1)]
    return data11, data12


def preprocess_data(df, seasons=[1, 2], train_perc=0.8):
    drop_cols = ["timestamp"]

    df = df.drop(drop_cols, axis=1)
    df = df[df["season"].isin(seasons)].reset_index(drop=True)

    df_train, df_test = train_test_split(df, train_size=train_perc, shuffle=False)

    # select columns to be normalized (all except id and season columns)
    norm_columns = [c for c in df_train.columns if c not in ['id'] + season_cols()]

    # normalize the data
    scaler = StandardScaler()
    df_train[norm_columns] = scaler.fit_transform(df_train[norm_columns])
    df_test[norm_columns] = scaler.transform(df_test[norm_columns])

    # cols to be used as input
    x_cols = [c for c in df_train.columns if c not in ['id', 'cnt']]

    # extract the time windows and separate RUL as the label
    win = SlidingWindowPanel(window_len=24,
                             unique_id_cols=['id'],
                             get_x=x_cols,
                             get_y=['cnt'],
                             horizon=0,
                             stride=1)

    win_train, y_train = win(df_train)
    win_test, y_test = win(df_test)

    return win_train, y_train, win_test, y_test


def save_tl_datasets(path="../Data"):
    """
    load the bike sharing dataset and split the first 3 seasons as source and last one as target datasets
    split each dataset in training and test data and save them in the Data directory
    the windowed version of the data is stored as (samples, timesteps, channels)
    """
    df11, df12 = load()
    src_dfs = preprocess_data(df11, seasons=[1,2,3], train_perc=0.8)
    tar_dfs = preprocess_data(df11, seasons=[4], train_perc=0.2)

    c22_nan_cols = None
    for name, dfs in zip(['src', 'tar'], [src_dfs, tar_dfs]):
        tr_x, tr_y, tst_x, tst_y = dfs
        c22_tr, c22_tst, c22_nan_cols = catch22(tr_x, tst_x, c22_nan_cols)
        np.savez(
            os.path.join(path, 'bike', name+"11.npz"),
            win_x_train=tr_x.transpose(0, 2, 1),
            man_x_train=extract_manual_features(tr_x),
            c22_x_train=c22_tr,
            y_train=tr_y,
            win_x_test=tst_x.transpose(0, 2, 1),
            man_x_test=extract_manual_features(tst_x),
            c22_x_test=c22_tst,
            y_test=tst_y
        )


if __name__ == '__main__':
    save_tl_datasets()
