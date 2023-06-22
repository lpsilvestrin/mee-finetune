import wandb

from algorithms.wann import train_wann
from data_utils.datasets import load_preproc_data


def sweep_vanilla_wann():
    config = dict(
        validation_split=0.1,
        # seed=list(range(5)),
        seed=4,
        mlp=dict(
          learning_rate=1e-3,
          epochs=20,
          batch_size=64,
          early_stop_patience=4,
          hidden=dict(
              n=2,
              units=100
          ),
          l2_reg=1e-4,
          loss_function='mse',
          dropout_rate=0.1
        ),
        # train_dataset=['src', 'tar1', 'tar2', 'tar3'])
        train_dataset='bpm10_tar',
        src_dataset='bpm10_src',
        C=0.5,
        pre_train=True
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
    wandb_init['config'] = config

    test_data_dict = dict()
    train_data_dict = dict()
    for name in [config.train_dataset, config.src_dataset]:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict['man_x_test'], data_dict['y_test'].reshape(-1, 1))
        train_data_dict[name] = (data_dict['man_x_train'], data_dict['y_train'].reshape(-1, 1))

    test_data_dict["test"] = test_data_dict[config.train_dataset]
    src_x, src_y = train_data_dict[config.src_dataset]
    tar_x, tar_y = train_data_dict[config.train_dataset]
    train_wann(src_x, src_y, tar_x, tar_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    sweep_vanilla_wann()