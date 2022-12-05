import numpy as np
from omegaconf import DictConfig
from sktime.transformations.panel.catch22 import Catch22

from utils.datasets import load_preproc_data


def prepare_data(config: DictConfig):
    label_test = 'y_test'
    label_train = 'y_train'
    if 'trunc_label' in config and config.trunc_label is True:
        label_test = 'trunc_' + label_test
        label_train = 'trunc_' + label_train

    if 'input_type' in config:
        feat_train = config.input_type+'_x_train'
        feat_test = config.input_type+'_x_test'
    else:
        feat_train = 'win_x_train'
        feat_test = 'win_x_test'

    test_data_dict = dict()
    for name in config.test_dataset:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict[feat_test], data_dict[label_test].reshape(-1,1).astype(np.float32))

    # in case of 2 training sets, concatenate them together
    train_datasets = config['train_dataset'].split('+')
    data_dict = load_preproc_data(name=train_datasets[0])
    win_x, win_y = data_dict[feat_train], data_dict[label_train].reshape(-1,1).astype(np.float32)

    if len(train_datasets) > 1:
        data_dict = load_preproc_data(name=train_datasets[1])
        win_x = np.concatenate([win_x, data_dict[feat_train]])
        win_y = np.concatenate([win_y, data_dict[label_train].reshape(-1,1).astype(np.float32)])

    test_data_dict["test"] = (data_dict[feat_test], data_dict[label_test].reshape(-1,1).astype(np.float32))

    if 'transpose_input' in config and config.transpose_input is True:
        for k, d in test_data_dict.items():
            x, y = d
            test_data_dict[k] = (tr(x), y)
        win_x = tr(win_x)

    return win_x, win_y, test_data_dict


def tr(x):
    return x.transpose(0, 2, 1)


def catch22(tr_x, tst_x, drop_cols=None):
    """
    expects inputs in the shape (samples, features, steps)
    Args:
        tr_x:
        tst_x:
        drop_cols:

    Returns:

    """
    c22 = Catch22(n_jobs=8)
    c22_tr = np.array(c22.fit_transform(tr_x))
    c22_tst = np.array(c22.transform(tst_x))

    if drop_cols is None:
        nan_cols = np.any(np.isnan(c22_tr), axis=0)
    else:
        nan_cols = drop_cols

    c22_tr = c22_tr[:, ~nan_cols]
    c22_tst = c22_tst[:, ~nan_cols]
    return c22_tr, c22_tst, nan_cols


def extract_manual_features(window_dataset):
    """
    given a multivariate time-window dataset, summarize over time all the variables in each window
    using min, max, mean and std statistics
    expects input in the format (samples, features, steps)
    Args:
        window_dataset:

    Returns:

    """
    mean_feat = np.mean(window_dataset, axis=2)
    std_feat = np.std(window_dataset, axis=2)
    min_feat = np.min(window_dataset, axis=2)
    max_feat = np.max(window_dataset, axis=2)
    return np.concatenate([mean_feat, std_feat, max_feat, min_feat], axis=1)
