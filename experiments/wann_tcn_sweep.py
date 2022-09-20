import wandb
from algorithms.wann import train_wann_tcn
from utils.nasa_data_preprocess import load_preproc_data


def sweep_wann_tcn():
    config = dict(
        C=0.1,
        pre_train=False,
        learning_rate=1e-4,
        dropout_rate=0.1,
        loss_function='mse',
        epochs=4,
        batch_size=64,
        validation_split=0.1,
        early_stop_patience=2,
        seed=4,
        tcn=dict(dilations=2,
                 filters=128),
        tcn2=True,
        kernel_size=3,
        transpose_input=True,
        save_model=False,
        l2_reg=1e-2,
        train_dataset='tar1'
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
    for name in [config.train_dataset, 'src']:
        data_dict = load_preproc_data(name=name)
        test_data_dict[name] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1, 1))
        train_data_dict[name] = (data_dict['win_x_train'], data_dict['y_train'].reshape(-1, 1))

    test_data_dict["test"] = test_data_dict[config.train_dataset]
    src_x, src_y = train_data_dict['src']
    tar_x, tar_y = train_data_dict[config.train_dataset]
    train_wann_tcn(src_x, src_y, tar_x, tar_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    sweep_wann_tcn()