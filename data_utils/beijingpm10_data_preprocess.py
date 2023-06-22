import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tsai.data.external import get_Monash_regression_data

from algorithms.utils import catch22, extract_manual_features


def load_train(data_path=""):
    X, y, _ = get_Monash_regression_data("BeijingPM10Quality", path="", split_data=False)


def preprocess_data(x_train, x_test, y_train, y_test, scaler=None):
    # flatten windows
    _, channels, win_size = x_train.shape
    x_train = x_train.transpose(0,2,1).reshape(-1, channels)
    x_test = x_test.transpose(0,2,1).reshape(-1, channels)

    # normalize the data per input channel using training samples
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # unflatten the time windows (samples, window size, channels)
    x_train = x_train.reshape(-1, win_size, channels)
    x_test = x_test.reshape(-1, win_size, channels)

    # normalize the outputs
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))

    return x_train, x_test, y_train, y_test


def save_tl_datasets(path="../Data/"):
    """
    load the beijing PM10 dataset and split the first 5000 samples as source and last 5000 as target datasets
    split each dataset in training and test data and save them in the Data directory
    the windowed version of the data is stored as (samples, timesteps, channels)

    """
    dsid = "BeijingPM10Quality"

    X, y, _ = get_Monash_regression_data(dsid, path=path, split_data=False)

    # remove nans
    nan_idx = np.any(np.isnan(X), axis=(1,2))
    X, y = X[~nan_idx], y[~nan_idx]

    src_x, src_y = X[:5000, :, :], y[:5000]
    tar_x, tar_y = X[-1200:, :, :], y[-1200:]
    src = train_test_split(src_x, src_y, test_size=1000, shuffle=False)
    tar = train_test_split(tar_x, tar_y, test_size=1000, shuffle=False)

    preproc_dfs = [('src', src), ('tar', tar)]
    c22_drop_cols = None
    for name, dfs in preproc_dfs:
        tr_x, tst_x, tr_y, tst_y = preprocess_data(*dfs)
        c22_tr, c22_tst, c22_cols = catch22(tr_x.transpose(0, 2, 1), tst_x.transpose(0, 2, 1), c22_drop_cols)
        np.savez(os.path.join(path, dsid, f"{name}.npz"),
                 win_x_train=tr_x,
                 man_x_train=extract_manual_features(tr_x.transpose(0, 2, 1)),
                 c22_x_train=c22_tr,
                 y_train=tr_y,
                 win_x_test=tst_x,
                 man_x_test=extract_manual_features(tst_x.transpose(0, 2, 1)),
                 c22_x_test=c22_tst,
                 y_test=tst_y)


if __name__ == '__main__':
    save_tl_datasets()