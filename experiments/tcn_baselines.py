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

from sklearn.model_selection import ParameterGrid

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def train_tcn(train_x, train_y, test_x, test_y, init_dict):
    run = wandb.init(**init_dict)

    config = wandb.config
    # config = init_dict['config']

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)

    nb_features = train_x.shape[2]
    nb_steps = train_x.shape[1]
    nb_out = 1

    i = Input(shape=(nb_steps, nb_features))
    #m = TCN()(i)
    #m = Dense(1, activation='linear')(m)

    m = TCN(nb_filters=32, dropout_rate=config.dropout_rate, dilations=[1, 2, 4, 8, 16, 32], return_sequences=True)(i)
    m = TCN(nb_filters=16, dropout_rate=config.dropout_rate, dilations=[1, 2, 4, 8, 16, 32], return_sequences=False)(m)
    m = Dense(nb_out, activation='linear')(m)
    model = Model(inputs=[i], outputs=[m])

    adam_opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    # adam_opt = 'adam'
    model.compile(loss=config.loss_function, optimizer=adam_opt, metrics=['mae', r2_keras, root_mean_squared_error])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
        restore_best_weights=True,
    )

    # callbacks = []
    callbacks = [early_stop]
    callbacks.append(WandbCallback())

    history = model.fit(train_x, train_y,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_split=config.validation_split,
                        verbose=2,
                        callbacks=callbacks)

    result = model.evaluate(test_x, test_y)
    test_metric_names = ["test_"+n for n in model.metrics_names]
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
    else:
        win_x = np.load('../Data/df1/swpX.npy')
        win_y = np.load('../Data/df1/swpy.npy')
        tst_x = np.load('../Data/df1/swpX_test.npy')
        tst_y = np.load('../Data/df1/swpy_test.npy')

        if not op_inputs:
            win_x = win_x[:, 2:, :]
            tst_x = tst_x[:, 2:, :]

    if transpose_input:
        win_x = np.transpose(win_x, (0, 2, 1))
        tst_x = np.transpose(tst_x, (0, 2, 1))

    train_tcn(win_x, win_y, tst_x, tst_y, wandb_init)


if __name__ == '__main__':

    wandb_init = {
        "project": 'debug_datasets', 'entity': 'transfer-learning-tcn', 'reinit': False,
        'config': {
            'learning_rate': [1e-3, 1e-2, 1e-4],
            'dropout_rate': [0.2],
            'loss_function': ['mse'],
            'epochs': [100],
            'batch_size': [200],
            'validation_split': [0.1],
            'early_stop_patience': [15],
            'seed': list(range(5)),
            'gijs': [True, False],
            'transpose_input': [False, True],
            'op_inputs': [False, True]
        }
    }

    grid = list(ParameterGrid(wandb_init['config']))

    for config in grid:
        wandb_init['config'] = config
        debug_datasets(wandb_init)



