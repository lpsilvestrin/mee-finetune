from itertools import product

import numpy as np
from omegaconf import DictConfig

import wandb
from algorithms.custom_loss_torch_lit_class import PyLitModelWrapper
from algorithms.torch_tcn import build_tcn
from data_utils.data_utils import prepare_data

from algorithms.custom_loss_torch_train import train_torch

_previous_src_paths = dict()


def run(loss_src='MEE', loss_tar='mse', seed=0, tar_data="tar1", src_data="src", last_only=False):
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
        model_type='tcn',
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

    train_x, train_y, test_data_dict = prepare_data(DictConfig(config))
    # test whether src was already trained before
    # create unique id for each src run
    src_id = "_".join([loss_src, src_data, str(seed)])
    if src_id not in _previous_src_paths:

        # pretain on the source data
        litmodel, ckpt_path = train_torch(train_x, train_y, test_data_dict, wandb_init)
        _previous_src_paths[src_id] = ckpt_path
    else:
        litmodel = load_pretrained_src(config, train_x, train_y, _previous_src_paths[src_id])

    # finetune on the target data
    group = tar_data
    if last_only:
        group += "_last_only"
    config['group'] = tar_data
    config['loss_function'] = loss_tar
    config['loss_src'] = loss_src
    config['train_dataset'] = tar_data
    # config['test_dataset'] = [tar_data]
    wandb_init['config'] = config

    if last_only:
        litmodel.model.tcn.requires_grad_(False)

    train_x, train_y, test_data_dict = prepare_data(DictConfig(config))

    train_torch(train_x, train_y, test_data_dict, wandb_init, model=litmodel.model)


def build_tcn_model(config, train_x, train_y):
    out_shape = train_y.shape[1]
    in_shape = train_x.shape[1:]
    in_features = in_shape[0]
    model = build_tcn(in_features, out_shape, config)
    return model


def load_pretrained_src(config, train_x, train_y, ckpt_path):
    model = build_tcn_model(DictConfig(config), train_x, train_y)
    litmodel = PyLitModelWrapper.load_from_checkpoint(ckpt_path, model=model)
    return litmodel


if __name__ == '__main__':
    data_pairs = []
    # CMPASS datasets
    data_pairs = data_pairs + list(product(['src'], ['tar1', 'tar2', 'tar3']))
    # data_pairs.append(('src', 'tar2'))
    data_pairs.append(('bpm10_src', 'bpm10_tar'))
    data_pairs.append(('bike11_src', 'bike11_tar'))
    # mee_pair = ['MEE', 'mse']
    # hsic_pair = ['HSIC', 'mse']
    # mae_pair = ['MAE', 'mse']
    all = ['MEE', 'mse', 'MAE', 'HSIC']
    # lpairs = list(product(mee_pair, mee_pair))
    lpairs = list(product(['mse'], all))
    # lpairs = list(product(mae_pair, mae_pair))
    # lpairs.remove(('MAE', 'MAE'))
    # lpairs = list(product(['HSIC'], hsic_pair))
    # lpairs.remove(('mse', 'mse'))
    # lpairs.append(('MAE', 'MAE'))
    last_layer = True
    params = product(lpairs, list(range(0, 20)), data_pairs)
    for loss_pair, seed, src_tar_pair in params:
        loss_src, loss_tar = loss_pair
        src, tar = src_tar_pair
        run(loss_src=loss_src, loss_tar=loss_tar, seed=seed, src_data=src, tar_data=tar, last_only=last_layer)
