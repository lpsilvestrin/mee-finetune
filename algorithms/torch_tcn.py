import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.nn.utils import weight_norm


class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, dilation, left_pad, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(ch_in, ch_out, kernel_size, stride=stride, padding=(left_pad, 0), dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(ch_out, ch_out, kernel_size, stride=stride, padding=(left_pad, 0), dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.short_net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)

        self.fit_shape = nn.Conv1d(ch_in, ch_out,
                                   kernel_size=1) if ch_in != ch_out else None  # Make sure we can add input and output of residual block
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        self.conv1.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.normal_(0, 0.01)

        if self.short_net is not None:
            self.fit_shape.weight.data.normal_(0, 0.01)
            self.fit_shape.bias.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.short_net(x)
        res = x if self.fit_shape == None else self.fit_shape(x)
        return self.relu(out + res)


class TCN_base(nn.Module):
    """
    Torch implementatino of TCN taken from https://github.com/samyakmshah/pytorchTCN/blob/master/tcn.py
    """
    def __init__(self, in_n, ch_n, out_n, kernel_size=2, dropout=0.2):
        super(TCN_base, self).__init__()
        layers = []
        lvl_n = len(ch_n)
        for i in range(lvl_n):
            dilation = 2 ** i
            ch_in = in_n if i == 0 else ch_n[i - 1]
            ch_out = ch_n[i]
            layers += [ResidualBlock(ch_in, ch_out, kernel_size, stride=1, dilation=dilation,
                                     left_pad=(kernel_size - 1) * dilation, dropout=dropout)]

        self.network = nn.Sequential(*layers)  # The * is to unpack the array
        self.fc = nn.Linear(ch_out, out_n)

    def forward(self, x):
        res = self.network(x)
        return self.fc(res[:,:,-1].squeeze(-1))


def build_tcn(
        nb_features: int,
        nb_out: int,
        config: DictConfig) -> nn.Module:

    return TCN_base(nb_features, config.filters, nb_out, config.kernel_size, config.dropout_rate)


def train_tcn(train_x, train_y, test_sets, wandb_init):
    run = wandb.init(**wandb_init)
    config = wandb.config

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
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

    model = build_tcn(nb_features, nb_steps, nb_out, config)

    adam_opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    # adam_opt = 'adam'
    rmse = tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')
    model.compile(loss=config.loss_function, optimizer=adam_opt, metrics=['mae', rmse])

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