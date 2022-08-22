import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from tsai.data.preparation import SlidingWindowPanel

def load(dataset_id, data_path="/home/luis/Documents/datasets/nasa_turbofan/"):
    """
    for a given id, load the corresponding train and test datasets;
    compute the RUL at each cycle for each engine and append it as a column in the final dataframes
    Args:
        dataset_id: X in 'FD00X' filenames
        data_path: path to the data files

    Returns: train and test dataframes with RUL

    """
    op_set = ["op" + str(i) for i in range(1, 4)]
    sensor = ["sensor" + str(i) for i in range(1, 22)]
    columns = ["id", "cycle"] + op_set + sensor

    # the usecols argument prevent pandas from interpreting the extra spaces in the end of each row as extra columns
    train = pd.read_csv(os.path.join(data_path, f"train_FD00{dataset_id}.txt"), sep=" ", usecols=list(range(len(columns))), names=columns)
    # compute RUL at each cycle of each engine in the train dataset
    max_cycles = train.groupby("id")['cycle'].transform('max')
    train["RUL"] = max_cycles - train['cycle']
    test = pd.read_csv(os.path.join(data_path, f"test_FD00{dataset_id}.txt"), sep=" ", usecols=list(range(len(columns))), names=columns)

    # compute RUL at each cycle of each engine in the test dataset
    max_cycles = test.groupby("id")['cycle'].transform('max')
    test["RUL"] = max_cycles - test['cycle']
    # compute nb of cycles for each engine in the test dataset
    test_cycles = test.groupby('id')['cycle'].max()
    # read the ground truth RUL at the last cycle
    true_rul = pd.read_csv(os.path.join(data_path, f"RUL_FD00{dataset_id}.txt"), sep=" ", usecols=[0], header=None)
    true_rul = true_rul[0].repeat(np.array(test_cycles)).reset_index(drop=True)
    # shift the RUL at each cycle by the ground truth RUL
    test["RUL"] = test["RUL"] + true_rul

    return train, test


def preprocess_data(df, nb_train_engines=80):
    """
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
    drop_cols = drop_cols + op_set

    df = df.drop(drop_cols, axis=1)

    # sample engines for training and for test
    ids = df["id"].unique()
    # create a random state for reproducibility
    rand = np.random.RandomState(1234)
    rand.shuffle(ids)
    df_train = df[df["id"].isin(ids[:nb_train_engines])]
    y_train = df_train.pop('RUL')
    df_test = df[df["id"].isin(ids[nb_train_engines:])]
    y_test = df_test.pop('RUL')
    # save input column names
    x_cols = df_train.columns

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    pt = PowerTransformer()
    df_train = scaler.fit_transform(df_train)
    df_train = pt.fit_transform(df_train)
    df_train = pd.DataFrame(df_train, columns=x_cols)
    df_train['RUL'] = y_train
    df_test = scaler.transform(df_test)
    df_test = pt.transform(df_test)
    df_test = pd.DataFrame(df_test, columns=x_cols)
    df_test['RUL'] = y_test

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


def extract_manual_features(window_dataset):
    """
    given a multivariate time-window dataset, summarize over time all the variables in each window
    using min, max, mean and std statistics
    Args:
        window_dataset:

    Returns:

    """
    mean_feat = np.mean(window_dataset, axis=2)
    std_feat = np.std(window_dataset, axis=2)
    min_feat = np.min(window_dataset, axis=2)
    max_feat = np.max(window_dataset, axis=2)
    return np.concatenate([mean_feat, std_feat, max_feat, min_feat], axis=1)


def save_tl_datasets():
    src_df, _ = load(1)
    # tar1_df, _ = load(2)
    # tar2_df, _ = load(3)
    # tar3_df, _ = load(4)
    src_tr_x, src_tr_y, src_tst_x, src_tst_y = preprocess_data(src_df, nb_train_engines=80)
    np.savez("../Data/df1/win_x.npz",
             win_x_train=src_tr_x,
             man_x_train=extract_manual_features(src_tr_x),
             y_train=src_tr_y,
             win_x_test=src_tst_x,
             man_x_test=extract_manual_features(src_tst_x),
             y_test=src_tst_y)






if __name__ == '__main__':
    # train, test = load(4)
    # print("train:", train.groupby('id')["cycle"].max().min())
    # print("test:", test.groupby('id')["cycle"].max().min())
    save_tl_datasets()
