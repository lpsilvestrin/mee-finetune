import numpy as np
import wandb

from utils.custom_loss_tcn import train_custom_loss_tcn
from utils.nasa_data_preprocess import load_preproc_data


def sweep_custom_tcn():
    # best TCN hyperparameters found using the source dataset
    config = dict(learning_rate=1e-6,
                  dropout_rate=0.1,
                  loss_function='MI',
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
                  train_dataset='src')

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
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1,1).astype(np.float32))

    wandb.init(**wandb_init)
    config = wandb.config
    print(config)
    # in case of 2 training sets, concatenate them together
    train_datasets = config['train_dataset'].split('+')
    data_dict = load_preproc_data(name=train_datasets[0])
    win_x, win_y = data_dict['win_x_train'], data_dict['y_train'].reshape(-1,1).astype(np.float32)

    if len(train_datasets) > 1:
        data_dict = load_preproc_data(name=train_datasets[1])
        win_x = np.concatenate([win_x, data_dict['win_x_train']])
        win_y = np.concatenate([win_y, data_dict['y_train'].reshape(-1,1).astype(np.float32)])

    test_data_dict["test"] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1,1).astype(np.float32))

    wandb_init['config'] = config
    train_custom_loss_tcn(win_x, win_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    sweep_custom_tcn()