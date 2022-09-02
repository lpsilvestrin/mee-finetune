import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

import random

from tcn import TCN

from utils.nasa_data_preprocess import load_preproc_data

import wandb
from wandb.keras import WandbCallback

from sklearn.model_selection import ParameterGrid

from utils.train_keras_tcn import train_tcn


def debug_datasets(wandb_init):
    transpose_input = wandb_init['config']['transpose_input']
    gijs = wandb_init['config']['gijs']
    op_inputs = wandb_init['config']['op_inputs']

    if not gijs:
        data_dict = load_preproc_data(name='src')
        win_x = data_dict['win_x_train']
        win_y = data_dict['y_train']
        tst_x = data_dict['win_x_test']
        tst_y = data_dict['y_test']

    if transpose_input:
        win_x = np.transpose(win_x, (0, 2, 1))
        tst_x = np.transpose(tst_x, (0, 2, 1))

    train_tcn(win_x, win_y, tst_x, tst_y, wandb_init)


def gridsearch_tcn_baseline():
    grid_tcn = list(ParameterGrid({
        "filters": [8, 16, 32],
        "dilations": [2, 3, 4]
    }))
    wandb_init = {
        "project": 'gridsearch_src_tcn', 'entity': 'transfer-learning-tcn', 'reinit': False,
        'config': {
            'learning_rate': [1e-2, 1e-3],
            'dropout_rate': [0.2, 0.1],
            'loss_function': ['mse'],
            'epochs': [100],
            'batch_size': [200, 64],
            'validation_split': [0.1],
            'early_stop_patience': [15],
            'seed': list(range(5)),
            'tcn': grid_tcn,
            'tcn2': [True, False],
            'kernel_size': [2, 3],
            'transpose_input': [True, False]
        }
    }

    grid = list(ParameterGrid(wandb_init['config']))

    data_dict = load_preproc_data(name='src')
    win_x = data_dict['win_x_train']
    win_y = data_dict['y_train']

    test_data_dict = {"test": (data_dict['win_x_test'], data_dict['y_test'])}
    for name in ["tar1", "tar2", "tar3"]:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'])

    for config in grid:
        wandb_init['config'] = config
        train_tcn(win_x, win_y, test_data_dict, wandb_init)


def wandb_sweep():
    config = {'learning_rate': 1e-2,
            'dropout_rate': 0.1,
            'loss_function': 'mse',
            'epochs': 4,
            'batch_size': 200,
            'validation_split': 0.1,
            'early_stop_patience': 15,
            'seed': 0,
            'tcn': {'dilations': 1,
                    'filters': 8},
            'tcn2': False,
            'kernel_size': 2,
            'transpose_input': True}
    wandb_init = dict(
        project='test2',
        entity='transfer-learning-tcn',
        reinit=False,
        config=config
    )
    data_dict = load_preproc_data(name='src')
    win_x = data_dict['win_x_train']
    win_y = data_dict['y_train']

    test_data_dict = {"test": (data_dict['win_x_test'], data_dict['y_test'])}
    for name in ["tar1", "tar2", "tar3"]:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'])

    train_tcn(win_x, win_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    wandb_sweep()




