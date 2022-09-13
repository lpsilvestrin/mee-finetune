import wandb
from algorithms.train_tradaboost import train_tradaboost_tcn

from utils.nasa_data_preprocess import load_preproc_data


def sweep_tradaboost():
    config = dict(lr=1,
                  loss_function='mse',
                  validation_split=0.1,
                  # seed=list(range(5)),
                  seed=4,
                  cv=3,
                  n_estimators=2,
                  n_estimators_fs=2,
                  src_run_path='transfer-learning-tcn/test2/17on4u6k',
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


if __name__ == '__main__':
    sweep_tradaboost()
