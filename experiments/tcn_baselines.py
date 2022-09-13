import numpy as np

from utils.nasa_data_preprocess import load_preproc_data

import wandb

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


def train_and_log_baselines():
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
                  save_model=False,
                  l2_reg=1e-2,
                  # train_dataset=['src', 'tar1', 'tar2', 'tar3'])
                  train_dataset='src+tar1')

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
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1,1))

    wandb.init(**wandb_init)
    config = wandb.config
    print(config)
    # in case of 2 training sets, concatenate them together
    train_datasets = config['train_dataset'].split('+')
    data_dict = load_preproc_data(name=train_datasets[0])
    win_x, win_y = data_dict['win_x_train'], data_dict['y_train'].reshape(-1,1)

    if len(train_datasets) > 1:
        data_dict = load_preproc_data(name=train_datasets[1])
        win_x = np.concatenate([win_x, data_dict['win_x_train']])
        win_y = np.concatenate([win_y, data_dict['y_train'].reshape(-1,1)])

    test_data_dict["test"] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1,1))

    wandb_init['config'] = config
    train_tcn(win_x, win_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    train_and_log_baselines()




