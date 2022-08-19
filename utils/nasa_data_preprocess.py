import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


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


if __name__ == '__main__':
    train, test = load(4)
    print("train:", train.groupby('id')["cycle"].max().min())
    print("test:", test.groupby('id')["cycle"].max().min())
