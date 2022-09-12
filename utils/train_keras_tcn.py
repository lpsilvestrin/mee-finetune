from types import SimpleNamespace
from typing import Union, TextIO

import numpy as np
from omegaconf import OmegaConf, DictConfig

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

import random

from algorithms.tcn import TCN

from wandb.keras import WandbCallback
import wandb

from sklearn.model_selection import train_test_split


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def build_tcn_from_config(nb_features: int, nb_steps: int, nb_out: int, config: DictConfig):
    i = Input(shape=(nb_steps, nb_features))

    return_sequences = True if config.tcn2 else False
    dilations = np.exp2(np.arange(config.tcn['dilations'])).astype('int').tolist()
    l2 = config.l2_reg if 'l2_reg' in config else 0
    l2_reg = keras.regularizers.L2(l2)
    m = TCN(kernel_size=config.kernel_size,
            nb_filters=config.tcn['filters'],
            dropout_rate=config.dropout_rate,
            dilations=dilations,
            return_sequences=return_sequences,
            kernel_regularizer=l2_reg)(i)
    if config.tcn2:
        m = TCN(kernel_size=config.kernel_size,
                nb_filters=config.tcn['filters'],
                dropout_rate=config.dropout_rate,
                dilations=dilations,
                kernel_regularizer=l2_reg,
                return_sequences=False)(m)
    m = Dense(nb_out,
              activation='linear',
              kernel_regularizer=l2_reg)(m)
    return Model(inputs=[i], outputs=[m])


def train_tcn(train_x, train_y, test_sets, wandb_init):
    run = wandb.init(**wandb_init)
    config = wandb.config

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)

    if config.transpose_input:
        train_x = np.transpose(train_x, (0, 2, 1))

    rand_state = np.random.RandomState(config.seed)
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y,
                                                test_size=config.validation_split,
                                                random_state=rand_state)

    nb_features = train_x.shape[2]
    nb_steps = train_x.shape[1]
    nb_out = 1

    model = build_tcn_from_config(nb_features, nb_steps, nb_out, config)

    adam_opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    # adam_opt = 'adam'
    model.compile(loss=config.loss_function, optimizer=adam_opt, metrics=['mae', r2_keras, root_mean_squared_error])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stop_patience,
        mode="min",
        restore_best_weights=True,
    )

    # callbacks = []
    callbacks = [early_stop]
    if 'save_model' in config.keys():
        save_model = config.save_model
    else:
        save_model = False
    callbacks.append(WandbCallback(save_model=save_model,
                                   save_graph=save_model))

    history = model.fit(tr_x, tr_y,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_data=(val_x, val_y),
                        verbose=2,
                        callbacks=callbacks)

    result = model.evaluate(val_x, val_y)
    test_metric_names = ["val/" + n for n in model.metrics_names]
    run.log(dict(zip(test_metric_names, result)))

    for key, t_set in test_sets.items():
        tst_x, tst_y = t_set
        if config.transpose_input:
            tst_x = np.transpose(tst_x, (0, 2, 1))
        result = model.evaluate(tst_x, tst_y)
        test_metric_names = [key + "/" + n for n in model.metrics_names]
        run.log(dict(zip(test_metric_names, result)))

    wandb.finish()

    return model


def restore_wandb_tcn_files(run_path: str) -> (DictConfig, Union[None, TextIO]):
    config_file = wandb.restore('config.yaml', run_path=run_path, replace=True)
    weight_file = wandb.restore('model-best.h5', run_path=run_path, replace=True)
    wandb_config = OmegaConf.load(config_file.name)

    del wandb_config['_wandb']
    del wandb_config['wandb_version']
    config = dict([(k, v['value']) for k, v in wandb_config.items()])
    config = OmegaConf.create(config)
    return config, weight_file


def finetune_tcn(train_x, train_y, test_sets, wandb_init):
    run = wandb.init(**wandb_init)
    config = wandb.config
    src_config, src_weight_file = restore_wandb_tcn_files(config.src_run_path)

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)

    if src_config.transpose_input:
        train_x = np.transpose(train_x, (0, 2, 1))

    rand_state = np.random.RandomState(config.seed)
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y,
                                                test_size=config.validation_split,
                                                random_state=rand_state)

    nb_features = train_x.shape[2]
    nb_steps = train_x.shape[1]
    nb_out = 1

    model = build_tcn_from_config(nb_features, nb_steps, nb_out, src_config)
    model.load_weights(src_weight_file.name)

    if config.last_layer:
        for layer in model.layers[:-1]:
            layer.trainable = False

    opt = keras.optimizers.get(config.optimizer)
    if config.optimizer == 'sgd':
        opt.nesterov = config.sgd_nesterov['enable']
        opt.momentum.assign(config.sgd_nesterov['momentum'])

    opt.learning_rate.assign(config.learning_rate)
    # adam_opt = 'adam'
    model.compile(loss=config.loss_function, optimizer=opt, metrics=['mae', r2_keras, root_mean_squared_error])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stop_patience,
        mode="min",
        restore_best_weights=True,
    )

    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=config.lr_schedule['factor'],
                                                    patience=config.lr_schedule['patience'],
                                                    verbose=1)

    # callbacks = []
    callbacks = [early_stop]
    if config.lr_schedule['enable']:
        callbacks.append(lr_sched)
    if 'save_model' in config.keys():
        save_model = config.save_model
    else:
        save_model = False
    callbacks.append(WandbCallback(save_model=save_model,
                                   save_graph=save_model))

    history = model.fit(tr_x, tr_y,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_data=(val_x, val_y),
                        verbose=2,
                        callbacks=callbacks)

    result = model.evaluate(val_x, val_y)
    test_metric_names = ["val/" + n for n in model.metrics_names]
    run.log(dict(zip(test_metric_names, result)))

    for key, t_set in test_sets.items():
        tst_x, tst_y = t_set
        if src_config.transpose_input:
            tst_x = np.transpose(tst_x, (0, 2, 1))
        result = model.evaluate(tst_x, tst_y)
        test_metric_names = [key + "/" + n for n in model.metrics_names]
        run.log(dict(zip(test_metric_names, result)))

    wandb.finish()

    return model