import pandas as pd
import os
import numpy as np


def load_train(data_path="/home/luis/datasets/beijing_air_quality/"):
    """
    load all the csv files from beijing air quality dataset and concatenate them on the same dataframe
    dataset link: https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data

    """
    file_list = os.listdir(data_path)
    dfs = [pd.read_csv(os.path.join(data_path, fn)) for fn in file_list if fn[-4:] == ".csv"]
    df = pd.concat(dfs)
    return df


def preprocess_data(df, nb_train_engines=80):
    """
    TODO 
    1 - drop sensors 1, 5, 6, 10, 16, 18, 19, operational settings and other columns not used for modelling
    2 - shuffle the engine ids and separate train and test sets
    3 - normalize the sensors based on the training observations
    4 - extract the time windows
    5 - drop id column
    6 - drop windows containing nan values
    7 - separate the last RUL value for each window as a label
    8 - drop the RUL column
    Args:
        df:
        nb_train_engines: total engines which will be sorted for training

    Returns:

    """

    op_set = ["op" + str(i) for i in range(1, 4)]
    sensor = np.array(["sensor" + str(i) for i in range(1, 22)])
    drop_cols = list(sensor[np.array([1,5,6,10,16,18,19])-1])
    drop_cols = drop_cols + op_set + ['cycle']

    df = df.drop(drop_cols, axis=1)

    # sample engines for training and for test
    ids = df["id"].unique()
    # create a random state for reproducibility
    rand = np.random.RandomState(1234)
    rand.shuffle(ids)
    df_train = df[df["id"].isin(ids[:nb_train_engines])].reset_index(drop=True)
    y_train = df_train.pop('RUL')
    id_train = df_train.pop('id')
    df_test = df[df["id"].isin(ids[nb_train_engines:])].reset_index(drop=True)
    y_test = df_test.pop('RUL')
    id_test = df_test.pop('id')
    # save input column names
    x_cols = df_train.columns

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    pt = PowerTransformer()
    df_train = scaler.fit_transform(df_train)
    df_train = pt.fit_transform(df_train)
    df_train = pd.DataFrame(df_train, columns=x_cols)
    df_train['RUL'] = y_train
    df_train['id'] = id_train
    df_test = scaler.transform(df_test)
    df_test = pt.transform(df_test)
    df_test = pd.DataFrame(df_test, columns=x_cols)
    df_test['RUL'] = y_test
    df_test['id'] = id_test

    # extract the time windows and separate RUL as the label
    win = SlidingWindowPanel(window_len=30,
                             unique_id_cols=['id'],
                             get_x=x_cols,
                             get_y=['RUL'],
                             horizon=0,
                             stride=1)

    win_train, rul_train = win(df_train)
    win_test, rul_test = win(df_test)

    return win_train, rul_train, win_test, rul_test