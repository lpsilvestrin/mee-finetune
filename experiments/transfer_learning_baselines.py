from types import SimpleNamespace

import numpy as np
import yaml
import random

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

import wandb
from wandb.keras import WandbCallback

from utils.nasa_data_preprocess import load_preproc_data
from utils.train_keras_tcn import build_tcn_from_config, r2_keras, root_mean_squared_error, finetune_tcn


def train_and_log_tl_baselines():
    # best TCN hyperparameters found using the source dataset
    config = dict(learning_rate=1e-3,
                  dropout_rate=0.1,
                  loss_function='mse',
                  epochs=4,
                  batch_size=64,
                  validation_split=0.1,
                  early_stop_patience=15,
                  # seed=list(range(5)),
                  seed=4,
                  tcn=dict(dilations=2,
                           filters=128),
                  tcn2=True,
                  kernel_size=3,
                  transpose_input=True,
                  save_model=True,
                  # train_dataset=['src', 'tar1', 'tar2', 'tar3'])
                  train_dataset='tar1')

    wandb_init = dict(
        project='test2',
        entity='transfer-learning-tcn',
        reinit=False,
        config=config
    )

    # grid = list(ParameterGrid(config))

    test_data_dict = dict()
    for name in ["tar1", "tar2", "tar3"]:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'])

    wandb.init(**wandb_init)
    config = wandb.config
    print(config)
    data_dict = load_preproc_data(name=config['train_dataset'])
    test_data_dict["test"] = (data_dict['win_x_test'], data_dict['y_test'])
    win_x, win_y = data_dict['win_x_train'], data_dict['y_train']
    wandb_init['config'] = config
    finetune_tcn(win_x, win_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    train_and_log_tl_baselines()