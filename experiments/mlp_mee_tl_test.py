from itertools import product

import numpy as np
from omegaconf import DictConfig

import wandb
from utils.data_utils import prepare_data

from utils.custom_loss_torch import train_torch

def run(loss_src='MEE', loss_tar='mse', seed=0):
    # test mlp hyperparameters
    config = dict(
        learning_rate=1e-3,
        dropout_rate=0.1,
        loss_function=loss_src,
        epochs=200,
        batch_size=64,
        validation_split=0.1,
        early_stop_patience=200,
        # seed=list(range(5)),
        seed=seed,
        hidden=[100, 100, 10],
        transpose_input=False,
        save_model=False,
        l2_reg=.0,
        train_dataset='src',
        test_dataset=['tar1'],
        norm_label=True,
        trunc_label=False,
        debug_mode=False,
        input_type='man',
        model_type='mlp'
    )

    wandb_init = dict(
        project='MEE_loss_test',
        entity='transfer-learning-tcn',
        group='nasa_tar1_src',
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
    wandb_init['group'] = 'nasa_tar1_tar'
    config['loss_function'] = loss_tar
    config['loss_src'] = loss_src
    config['train_dataset'] = 'tar1'

    train_x, train_y, test_data_dict = prepare_data(DictConfig(config))

    train_torch(train_x, train_y, test_data_dict, wandb_init, model=litmodel.model)


if __name__ == '__main__':
    params = product(['MEE', 'mse', 'MI'], ['mse'], list(range(0,10)))
    for loss_src, loss_tar, seed in params:
        run(loss_src=loss_src, loss_tar=loss_tar, seed=seed)
