import numpy as np
import wandb

from utils.custom_loss_keras_model import train_custom_loss_tcn
from utils.datasets import load_preproc_data


def sweep_custom_keras_model():
    # best TCN hyperparameters found using the source dataset
    # config = dict(
    #     learning_rate=1e-4,
    #     dropout_rate=0.1,
    #     loss_function='mse',
    #     epochs=4,
    #     batch_size=64,
    #     validation_split=0.1,
    #     early_stop_patience=15,
    #     # seed=list(range(5)),
    #     seed=3,
    #     tcn=dict(dilations=2,
    #            filters=128),
    #     tcn2=True,
    #     kernel_size=3,
    #     # transpose_input=False,
    #     save_model=False,
    #     l2_reg=0.01,
    #     train_dataset='src',
    #     test_dataset=['tar1'],
    #     trunc_label=False,
    #     debug_mode=False,
    #     input_type='win',
    #     model_type='tcn'
    # )

    # test mlp hyperparameters
    config = dict(
        learning_rate=1e-4,
        dropout_rate=0.1,
        loss_function='MEE',
        epochs=400,
        batch_size=64,
        validation_split=0.1,
        early_stop_patience=400,
        # seed=list(range(5)),
        seed=3,
        hidden=[100, 20],
        transpose_input=False,
        save_model=False,
        l2_reg=1e-5,
        train_dataset='src',
        test_dataset=[],
        trunc_label=False,
        debug_mode=False,
        input_type='man',
        model_type='mlp'
    )

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
    train_custom_loss_tcn(win_x, win_y, test_data_dict, wandb_init)


def prepare_data(config):
    label_test = 'y_test'
    label_train = 'y_train'
    if config.trunc_label:
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


if __name__ == '__main__':
    sweep_custom_keras_model()
