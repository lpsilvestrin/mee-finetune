import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

import random

from tcn import TCN

from utils.nasa_data_preprocess import load_preproc_data

import wandb
from wandb.keras import WandbCallback

from sklearn.model_selection import ParameterGrid, train_test_split

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def train_tcn(train_x, train_y, test_sets, init_dict):
    run = wandb.init(**init_dict)

    config = wandb.config
    # config = init_dict['config']

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)

    test_dict = {}
    if config.transpose_input:
        train_x = np.transpose(train_x, (0, 2, 1))

    rand_state = np.random.RandomState(config.seed)
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y,
                                                test_size=config.validation_split,
                                                random_state=rand_state)

    nb_features = train_x.shape[2]
    nb_steps = train_x.shape[1]
    nb_out = 1

    i = Input(shape=(nb_steps, nb_features))

    return_sequences = False if config.tcn2 is None else True
    dilations1 = np.exp2(np.arange(config.tcn1['dilations'])).astype('int').tolist()
    # dilations1 = [1,2]
    m = TCN(kernel_size=config.kernel_size, nb_filters=config.tcn1['filters'], dropout_rate=config.dropout_rate, dilations=dilations1, return_sequences=return_sequences)(i)
    if config.tcn2 is not None:
        dilations2 = np.exp2(np.arange(config.tcn2['dilations'])).astype('int').tolist()
        m = TCN(kernel_size=config.kernel_size, nb_filters=config.tcn2['filters'], dropout_rate=config.dropout_rate, dilations=dilations2, return_sequences=False)(m)
    m = Dense(nb_out, activation='linear')(m)
    model = Model(inputs=[i], outputs=[m])

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
    callbacks.append(WandbCallback())

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
        test_metric_names = [key+"/" + n for n in model.metrics_names]
        run.log(dict(zip(test_metric_names, result)))

    run.join()
    wandb.finish()


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
        "project": 'test2', 'entity': 'transfer-learning-tcn', 'reinit': False,
        'config': {
            'learning_rate': [1e-2, 1e-3],
            'dropout_rate': [0.2, 0.1],
            'loss_function': ['mse'],
            'epochs': [4],
            'batch_size': [200, 64],
            'validation_split': [0.1],
            'early_stop_patience': [15],
            'seed': list(range(5)),
            'tcn1': grid_tcn,
            'tcn2': [None] + grid_tcn,
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

    for config in grid[:2]:
        wandb_init['config'] = config
        train_tcn(win_x, win_y, test_data_dict, wandb_init)


if __name__ == '__main__':
    gridsearch_tcn_baseline()




