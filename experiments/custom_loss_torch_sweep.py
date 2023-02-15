import numpy as np
from omegaconf import DictConfig

import wandb
from utils.data_utils import prepare_data

from utils.custom_loss_keras_model import train_custom_loss_keras_model
from utils.custom_loss_torch import train_torch
from utils.datasets import load_preproc_data


def sweep_custom_torch_model():
    config = dict()

    # best TCN hyperparameters found using the source dataset
    # config = dict(
    #     learning_rate=1e-4,
    #     dropout_rate=0.1,
    #     loss_function='MEE',
    #     sigma_y=1,
    #     sigma_x=2,
    #     epochs=100,
    #     batch_size=64,
    #     validation_split=0.1,
    #     early_stop_patience=10,
    #     early_stop_criteria='val_loss',
    #     # seed=list(range(5)),
    #     seed=7,
    #     tcn=dict(dilations=2,
    #              filters=128),
    #     tcn2=False,
    #     # filters=[128, 128],
    #     kernel_size=3,
    #     transpose_input=True,
    #     save_model=False,
    #     l2_reg=1e-5,
    #     train_dataset='src',
    #     test_dataset=[],
    #     debug_mode=False,
    #     input_type='win',
    #     model_type='tcn'
    # )

    # test mlp hyperparameters
    # config = dict(
    #     learning_rate=1e-4,
    #     dropout_rate=0.1,
    #     loss_function='MEE',
    #     epochs=400,
    #     batch_size=32,
    #     validation_split=0.1,
    #     early_stop_patience=400,
    #     # seed=list(range(5)),
    #     seed=3,
    #     hidden=[],
    #     transpose_input=False,
    #     save_model=False,
    #     l2_reg=1e-5,
    #     train_dataset='src',
    #     test_dataset=[],
    #     trunc_label=False,
    #     debug_mode=False,
    #     input_type='man',
    #     model_type='mlp'
    # )

    wandb_init = dict(
        project='test2',
        entity='transfer-learning-tcn',
        reinit=False,
        config=config
    )

    wandb.init(**wandb_init)
    config = wandb.config
    print(config)

    win_x, win_y, test_data_dict = prepare_data(config)

    wandb_init['config'] = config
    # train_custom_loss_tcn(win_x, win_y, test_data_dict, wandb_init)
    train_torch(win_x, win_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    sweep_custom_torch_model()
