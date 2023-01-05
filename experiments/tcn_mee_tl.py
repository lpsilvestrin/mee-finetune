from itertools import product

import numpy as np
from omegaconf import DictConfig

import wandb
from utils.data_utils import prepare_data

from utils.custom_loss_torch import train_torch


def run(loss_src='MEE', loss_tar='mse', seed=0):
    # test mlp hyperparameters
    config = dict(
        learning_rate=1e-4,
        dropout_rate=0.1,
        epochs=200,
        batch_size=64,
        validation_split=0.1,
        early_stop_patience=100,
        # seed=list(range(5)),
        filters=[128, 128],
        kernel_size=3,
        tcn2=False,
        seed=seed,
        transpose_input=True,
        save_model=False,
        l2_reg=.0,
        test_dataset=[],
        # norm_label=True,
        # trunc_label=False,
        debug_mode=False,
        input_type='win',
        model_type='tcn'
    )

    config['group'] = 'nasa_tar1_src'
    config['loss_function'] = loss_src
    config['train_dataset'] = 'src'

    wandb_init = dict(
        project='tcn_custom_loss',
        entity='transfer-learning-tcn',
        reinit=False,
        config=config
    )

    # wandb.init(**wandb_init)
    # config = wandb.config
    # print(config)

    train_x, train_y, test_data_dict = prepare_data(DictConfig(config))

    # pretain on the source data
    litmodel = train_torch(train_x, train_y, test_data_dict, wandb_init)

    # finetune on the target data
    config['group'] = 'nasa_tar1_tar'
    config['loss_function'] = loss_tar
    config['loss_src'] = loss_src
    config['train_dataset'] = 'tar1'
    config['test_dataset'] = ['tar1']
    wandb_init['config'] = config

    train_x, train_y, test_data_dict = prepare_data(DictConfig(config))

    train_torch(train_x, train_y, test_data_dict, wandb_init, model=litmodel.model)


if __name__ == '__main__':
    params = product(['MEE', 'mse'], ['MEE', 'mse'], list(range(0,20)))
    for loss_src, loss_tar, seed in params:
        run(loss_src=loss_src, loss_tar=loss_tar, seed=seed)