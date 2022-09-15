import random
import numpy as np
import tensorflow as tf
import keras

import wandb
from adapt.instance_based import WANN
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

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

    model = build_mlp(nb_features, nb_out, mlp_config)

    opt = keras.optimizers.Adam(learning_rate=mlp_config.learning_rate)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=mlp_config.early_stop_patience,
        mode="min",
        restore_best_weights=True,
    )

    # callbacks = []
    callbacks = [early_stop]

    wann = WANN(task=model,
                weighter=model,
                Xt=tr_x,
                yt=tr_y)

    metrics = [tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
               tf.keras.metrics.MeanAbsoluteError(name='mae'),
               tf.keras.metrics.MeanSquaredError(name='loss')]

    wann.compile(loss=mlp_config.loss_function, optimizer=opt, metrics={'disc': [], 'task': metrics})

    wann.fit(src_x, src_y,
             epochs=mlp_config.epochs,
             batch_size=mlp_config.batch_size,
             callbacks=callbacks,
             verbose=1)

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

    return model