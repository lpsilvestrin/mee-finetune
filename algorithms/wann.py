import random
import numpy as np
import tensorflow as tf
import keras

import wandb
from adapt.instance_based import WANN
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from utils.train_keras_tcn import build_tcn_from_config
from utils.utils import evaluate, build_mlp


def train_wann(src_x, src_y, tar_x, tar_y, test_sets, wandb_init):
    run = wandb.init(**wandb_init)
    config = wandb.config
    mlp_config = DictConfig(config.mlp)

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)

    rand_state = np.random.RandomState(config.seed)
    tr_x, val_x, tr_y, val_y = train_test_split(tar_x, tar_y,
                                                test_size=config.validation_split,
                                                random_state=rand_state)

    nb_features = tr_x.shape[1]
    nb_out = 1

    task_model = build_mlp(nb_features, nb_out, mlp_config)
    weighter_model = build_mlp(nb_features, nb_out, mlp_config, last_activation='relu')

    opt = keras.optimizers.Adam(learning_rate=mlp_config.learning_rate)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=mlp_config.early_stop_patience,
        mode="min",
        min_delta=1e-5,
        restore_best_weights=True,
    )

    # callbacks = []
    callbacks = [early_stop]

    metrics = [tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
               tf.keras.metrics.MeanAbsoluteError(name='mae'),
               tf.keras.metrics.MeanSquaredError(name='loss')]

    wann = WANN(task=task_model,
                weighter=weighter_model,
                Xt=tr_x,
                yt=tr_y,
                C=config.C,
                pretrain=config.pre_train,
                loss=mlp_config.loss_function,
                optimizer=opt,
                metrics={'disc': [], 'task': metrics})

    wann.fit(src_x, src_y,
             epochs=mlp_config.epochs,
             batch_size=mlp_config.batch_size,
             callbacks=callbacks,
             verbose=1,
             validation_data=(val_x, val_y))

    metrics_names = [m.name for m in metrics]

    result = evaluate(val_x, val_y, wann, metrics)
    test_metric_names = ["val/" + n for n in metrics_names]
    run.log(dict(zip(test_metric_names, result)))

    for key, t_set in test_sets.items():
        tst_x, tst_y = t_set
        result = evaluate(tst_x, tst_y, wann, metrics)
        test_metric_names = [key + "/" + n for n in metrics_names]
        run.log(dict(zip(test_metric_names, result)))

    wandb.finish()

    return wann


def train_wann_tcn(src_x, src_y, tar_x, tar_y, test_sets, wandb_init):
    run = wandb.init(**wandb_init)
    config = wandb.config

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)

    if config.transpose_input:
        src_x = np.transpose(src_x, (0, 2, 1))
        tar_x = np.transpose(tar_x, (0, 2, 1))

    rand_state = np.random.RandomState(config.seed)
    tr_x, val_x, tr_y, val_y = train_test_split(tar_x, tar_y,
                                                test_size=config.validation_split,
                                                random_state=rand_state)

    nb_features = tar_x.shape[2]
    nb_steps = tar_x.shape[1]
    nb_out = 1
    weighter_model = build_tcn_from_config(nb_features, nb_steps, nb_out, config, last_activation='relu')
    task_model = build_tcn_from_config(nb_features, nb_steps, nb_out, config, last_activation='linear')

    opt = keras.optimizers.Adam(learning_rate=config.learning_rate)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stop_patience,
        mode="min",
        min_delta=1e-5,
        restore_best_weights=True,
    )

    # callbacks = []
    callbacks = [early_stop]

    metrics = [tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
               tf.keras.metrics.MeanAbsoluteError(name='mae'),
               tf.keras.metrics.MeanSquaredError(name='loss')]

    # wann.compile(loss=config.loss_function, optimizer=opt, metrics={'disc': [], 'task': metrics})

    wann = WANN(task=task_model,
                weighter=weighter_model,
                Xt=tr_x,
                yt=tr_y,
                C=config.C,
                pretrain=config.pre_train,
                loss=config.loss_function,
                optimizer=opt,
                metrics={'disc': [], 'task': metrics})

    wann.fit(src_x, src_y,
             epochs=config.epochs,
             batch_size=config.batch_size,
             callbacks=callbacks,
             validation_data=(val_x, val_y),
             verbose=1)

    metrics_names = [m.name for m in metrics]

    result = evaluate(val_x, val_y, wann, metrics)
    test_metric_names = ["val/" + n for n in metrics_names]
    run.log(dict(zip(test_metric_names, result)))

    for key, t_set in test_sets.items():
        tst_x, tst_y = t_set
        if config.transpose_input:
            tst_x = np.transpose(tst_x, (0, 2, 1))
        result = evaluate(tst_x, tst_y, wann, metrics)
        test_metric_names = [key + "/" + n for n in metrics_names]
        run.log(dict(zip(test_metric_names, result)))

    wandb.finish()

    return wann