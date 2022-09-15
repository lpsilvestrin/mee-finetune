import numpy as np
import tensorflow as tf
import random
import keras

import wandb
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from adapt.instance_based import TwoStageTrAdaBoostR2

from utils.train_keras_tcn import (
    restore_wandb_config,
    build_tcn_from_config
)

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout


def train_tradaboost_tcn(src_x, src_y, tar_x, tar_y, test_sets, wandb_init):
    run = wandb.init(**wandb_init)
    config = wandb.config
    src_config = restore_wandb_config(config.src_run_path)

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)

    if src_config.transpose_input:
        src_x = np.transpose(src_x, (0, 2, 1))
        tar_x = np.transpose(tar_x, (0, 2, 1))

    rand_state = np.random.RandomState(config.seed)
    tr_x, val_x, tr_y, val_y = train_test_split(tar_x, tar_y,
                                                test_size=config.validation_split,
                                                random_state=rand_state)

    nb_features = tr_x.shape[2]
    nb_steps = tr_x.shape[1]
    nb_out = 1
    src_config.tcn2 = False
    src_config.tcn.nb_filters = 4
    src_config.tcn.dilations = 1
    model = build_tcn_from_config(nb_features, nb_steps, nb_out, src_config)

    opt = keras.optimizers.Adam(lr=src_config.learning_rate)
    model.compile(loss=src_config.loss_function, optimizer=opt)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=src_config.early_stop_patience,
        mode="min",
        restore_best_weights=True,
    )

    # callbacks = []
    callbacks = [early_stop]
    # if 'save_model' in config.keys():
    #     save_model = config.save_model
    # else:
    #     save_model = False
    # callbacks.append(WandbCallback(save_model=save_model,
    #                                save_graph=save_model))

    tradaboost = TwoStageTrAdaBoostR2(estimator=model,
                                      Xt=tr_x,
                                      yt=tr_y,
                                      lr=config.lr,
                                      cv=config.cv,
                                      copy=False,
                                      n_estimators=config.n_estimators,
                                      n_estimators_fs=config.n_estimators_fs,
                                      random_state=config.seed)

    tradaboost.fit(src_x, src_y,
                   epochs=src_config.epochs,
                   batch_size=src_config.batch_size,
                   callbacks=callbacks,
                   verbose=1)

    metrics = [tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
               tf.keras.metrics.MeanAbsoluteError(name='mae'),
               tf.keras.metrics.MeanSquaredError(name='loss')]
    metrics_names = [m.name for m in metrics]

    result = evaluate(val_x, val_y, tradaboost, metrics)
    test_metric_names = ["val/" + n for n in metrics_names]
    run.log(dict(zip(test_metric_names, result)))

    for key, t_set in test_sets.items():
        tst_x, tst_y = t_set
        if src_config.transpose_input:
            tst_x = np.transpose(tst_x, (0, 2, 1))
        result = evaluate(tst_x, tst_y, tradaboost, metrics)
        test_metric_names = [key + "/" + n for n in metrics_names]
        run.log(dict(zip(test_metric_names, result)))

    wandb.finish()

    return model


def train_tradaboost_nn(src_x, src_y, tar_x, tar_y, test_sets, wandb_init):
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

    model = build_mlp(nb_features, nb_out, mlp_config)

    opt = keras.optimizers.Adam(lr=mlp_config.learning_rate)
    model.compile(loss=mlp_config.loss_function, optimizer=opt)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=mlp_config.early_stop_patience,
        mode="min",
        restore_best_weights=True,
    )

    # callbacks = []
    callbacks = [early_stop]
    # if 'save_model' in config.keys():
    #     save_model = config.save_model
    # else:
    #     save_model = False
    # callbacks.append(WandbCallback(save_model=save_model,
    #                                save_graph=save_model))

    tradaboost = TwoStageTrAdaBoostR2(estimator=model,
                                      Xt=tr_x,
                                      yt=tr_y,
                                      cv=config.cv,
                                      n_estimators=config.n_estimators,
                                      n_estimators_fs=config.n_estimators_fs,
                                      random_state=config.seed)

    tradaboost.fit(src_x, src_y,
                   epochs=mlp_config.epochs,
                   batch_size=mlp_config.batch_size,
                   callbacks=callbacks,
                   verbose=1)

    metrics = [tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
               tf.keras.metrics.MeanAbsoluteError(name='mae'),
               tf.keras.metrics.MeanSquaredError(name='loss')]
    metrics_names = [m.name for m in metrics]

    result = evaluate(val_x, val_y, tradaboost, metrics)
    test_metric_names = ["val/" + n for n in metrics_names]
    run.log(dict(zip(test_metric_names, result)))

    for key, t_set in test_sets.items():
        tst_x, tst_y = t_set
        result = evaluate(tst_x, tst_y, tradaboost, metrics)
        test_metric_names = [key + "/" + n for n in metrics_names]
        run.log(dict(zip(test_metric_names, result)))

    wandb.finish()

    return model


def evaluate(x, y, model, metrics):
    pred = model.predict(x)
    result = []
    for met in metrics:
        met.update_state(y, pred)
        result.append(met.result().numpy())
        met.reset_state()
    return result


def build_mlp(nb_features: int, nb_out: int, config: DictConfig):
    i = Input(shape=(nb_features))

    l2 = config.l2_reg if 'l2_reg' in config else 0
    l2_reg = keras.regularizers.L2(l2)

    m = i
    for n in config.hidden:
        m = Dense(n,
                  activation='relu',
                  kernel_regularizer=l2_reg)(m)
        Dropout(config.dropout_rate)(m)

    m = Dense(nb_out,
              activation='linear',
              kernel_regularizer=l2_reg)(m)
    return Model(inputs=[i], outputs=[m])