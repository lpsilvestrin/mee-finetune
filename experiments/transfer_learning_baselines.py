import wandb

from utils.datasets import load_preproc_data
from utils.train_keras_tcn import finetune_tcn


def train_and_log_tl_baselines():
    config = dict(learning_rate=1e-5,
                  loss_function='mse',
                  epochs=4,
                  batch_size=64,
                  validation_split=0.1,
                  early_stop_patience=15,
                  lr_schedule=dict(
                      enable=True,
                      factor=0.2,
                      patience=8
                  ),
                  optimizer='adam',
                  sgd_nesterov=dict(
                      enable=False,
                      momentum=0.9
                  ),
                  # seed=list(range(5)),
                  seed=4,
                  save_model=False,
                  last_layer=True,
                  l2_reg=1e-5,
                  src_run_path='transfer-learning-tcn/non_tl_baselines/jf791kuq',
                  # train_dataset=['src', 'tar1', 'tar2', 'tar3'])
                  train_dataset='tar1')

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
    data_dict = load_preproc_data(name=config['train_dataset'])
    test_data_dict["test"] = (data_dict['win_x_test'], data_dict['y_test'].reshape(-1,1))
    win_x, win_y = data_dict['win_x_train'], data_dict['y_train'].reshape(-1,1)
    wandb_init['config'] = config
    finetune_tcn(win_x, win_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    train_and_log_tl_baselines()