import wandb
from algorithms.train_tradaboost import train_tradaboost_tcn, train_tradaboost_conv

from utils.datasets import load_preproc_data


# @profile
def sweep_tradaboost_tcn():
    config = dict(lr=1,
                  validation_split=0.1,
                  # seed=list(range(5)),
                  seed=4,
                  cv=1,
                  n_estimators=2,
                  n_estimators_fs=2,
                  src_run_path='transfer-learning-tcn/test2/2mjc4t5x',
                  # train_dataset=['src', 'tar1', 'tar2', 'tar3'])
                  train_dataset='tar1')

    wandb_init = dict(
        project='test2',
        entity='transfer-learning-tcn',
        reinit=False,
        config=config
    )

    wandb.init(**wandb_init)
    config = wandb.config
    print(config)
    wandb_init['config'] = config

    test_data_dict = dict()
    train_data_dict = dict()
    for name in [config.train_dataset, 'src']:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1,1))
        train_data_dict[name] = (data_dict['win_x_train'], data_dict['y_train'].reshape(-1,1))

    test_data_dict["test"] = test_data_dict[config.train_dataset]
    src_x, src_y = train_data_dict['src']
    tar_x, tar_y = train_data_dict[config.train_dataset]
    train_tradaboost_tcn(src_x, src_y, tar_x, tar_y, test_data_dict, wandb_init)


def sweep_tradaboost_conv():
    config = dict(lr=1,
                  learning_rate=1e-3,
                  validation_split=0.1,
                  transpose_input=True,
                  # seed=list(range(5)),
                  seed=4,
                  cv=1,
                  n_estimators=2,
                  n_estimators_fs=2,
                  conv=dict(
                      dilations=3,
                      filters=128,
                      l2_reg=1e-5,
                      kernel_size=3,
                      dropout_rate=0.1
                  ),
                  batch_size=64,
                  epochs=3,
                  loss_function='mse',
                  early_stop_patience=15,
                  # train_dataset=['src', 'tar1', 'tar2', 'tar3'])
                  train_dataset='tar1')

    wandb_init = dict(
        project='test2',
        entity='transfer-learning-tcn',
        reinit=False,
        config=config
    )

    wandb.init(**wandb_init)
    config = wandb.config
    print(config)
    wandb_init['config'] = config

    test_data_dict = dict()
    train_data_dict = dict()
    for name in [config.train_dataset, 'src']:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1,1))
        train_data_dict[name] = (data_dict['win_x_train'], data_dict['y_train'].reshape(-1,1))

    test_data_dict["test"] = test_data_dict[config.train_dataset]
    src_x, src_y = train_data_dict['src']
    tar_x, tar_y = train_data_dict[config.train_dataset]
    train_tradaboost_conv(src_x, src_y, tar_x, tar_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    sweep_tradaboost_conv()
