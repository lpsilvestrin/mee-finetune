from itertools import product

import numpy as np
from omegaconf import DictConfig

import wandb
from utils.data_utils import prepare_data

from utils.custom_loss_torch import train_torch


def run(loss_src='MEE', loss_tar='mse', seed=0, tar_data="tar1", src_data="src"):
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

    config['group'] = src_data
    config['loss_function'] = loss_src
    config['train_dataset'] = src_data

    wandb_init = dict(
        project='tcn_custom_loss_residuals',
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
    config['group'] = tar_data
    config['loss_function'] = loss_tar
    config['loss_src'] = loss_src
    config['train_dataset'] = tar_data
    # config['test_dataset'] = [tar_data]
    wandb_init['config'] = config

    train_x, train_y, test_data_dict = prepare_data(DictConfig(config))

    train_torch(train_x, train_y, test_data_dict, wandb_init, model=litmodel.model)


if __name__ == '__main__':
    data_pairs = product(['src'], ['tar1', 'tar2', 'tar3'])
    data_pairs.append(['bpm10_src', 'bpm10_tar'])
    params = product(['MEE', 'mse'], ['MEE', 'mse'], list(range(0,1)), data_pairs)
    for loss_src, loss_tar, seed, src_tar_pair in params:
        src, tar = src_tar_pair
        run(loss_src=loss_src, loss_tar=loss_tar, seed=seed, src_data=src, tar_data=tar)