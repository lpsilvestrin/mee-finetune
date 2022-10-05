import keras
import numpy as np
import tensorflow as tf
import random

import wandb
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbCallback

from utils.nasa_data_preprocess import normalize_label
from utils.train_keras_tcn import build_tcn_from_config


def train_custom_loss_tcn(train_x, train_y, test_sets, wandb_init):
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
    tr_y = normalize_label(tr_y, mode='in')

    nb_features = train_x.shape[2]
    nb_steps = train_x.shape[1]
    nb_out = 1

    model = build_tcn_from_config(nb_features, nb_steps, nb_out, config)

    loss_tracker = keras.metrics.Mean(name="loss")

    debug_mode = False
    if 'debug_mode' in config:
        debug_mode = config.debug_mode

    class CustomLossModel(keras.Model):
        """
        based on the CustomModel class from:
        https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel
        """
        def train_step(self, data):
            x, y = data

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute our own loss
                loss = loss_fn(x, y_pred, y, config.loss_function)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            if debug_mode:
                print()
                print(f"%%%%%%%%%%%%%% step {self.step_counter}")
                print(f"loss: {loss}")
                print(f"grad min: {tf.reduce_min(gradients[0])}")
                print(f"grad max: {tf.reduce_max(gradients[0])}")
                print(f"grad mean: {tf.reduce_mean(gradients[0])}")
                print(f"mae: {tf.reduce_mean(tf.abs(y_pred - y))}")
                self.step_counter += 1

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Compute our own metrics
            loss_tracker.update_state(loss)

            self.compiled_metrics.update_state(y, y_pred)
            result = {m.name: m.result() for m in self.metrics}
            result['loss'] = loss_tracker.result()

            return result

        def test_step(self, data):
            # Unpack the data
            x, y = data
            # Compute predictions
            y_pred = self(x, training=False)
            y_pred = normalize_label(y_pred, mode='out')
            # Updates the metrics tracking the loss
            # self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # Update the metrics.
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            return {m.name: m.result() for m in self.metrics}

    # Construct an instance of CustomModel
    model = CustomLossModel(model.inputs, model.outputs)
    model.step_counter = 0

    adam_opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    # adam_opt = 'adam'
    # rmse = tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')
    # mse = tf.keras.metrics.MeanSquaredError(name='mse')
    model.compile(optimizer=adam_opt, metrics=['mae', 'mse'], run_eagerly=debug_mode)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_mse",
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


def pairwise_distances(x: tf.Tensor):
    # x should be two dimensional
    instances_norm = tf.reduce_sum(x ** 2, -1, keepdims=True)
    return -2 * tf.matmul(x, tf.transpose(x)) + instances_norm + tf.transpose(instances_norm)


def calculate_gram_mat(x: tf.Tensor, sigma: float):
    dist = pairwise_distances(x)
    return tf.exp(-dist / sigma)


def tf_log2(x: tf.Tensor):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator


def reyi_entropy(x: tf.Tensor, sigma: float):
    alpha = 1.001
    k = calculate_gram_mat(x, sigma)
    k = k / tf.linalg.trace(k)
    eigv = tf.abs(tf.linalg.eig(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * tf_log2(tf.reduce_sum(eig_pow))
    return entropy


def joint_entropy(x: tf.Tensor, y: tf.Tensor, s_x: float, s_y: float):
    alpha = 1.001
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = tf.matmul(x, y)
    k = k / tf.linalg.trace(k)
    eigv = tf.abs(tf.linalg.eig(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * tf_log2(tf.reduce_sum(eig_pow))

    return entropy


def calculate_MI(x: tf.Tensor, y: tf.Tensor, s_x: float, s_y: float):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    normalize = Ixy / (tf.maximum(Hx, Hy) + 1e-16)
    return normalize
    # return Ixy


def GaussianKernelMatrix(x: tf.Tensor, sigma: float):
    pairwise_distances_ = pairwise_distances(x)
    return tf.exp(-pairwise_distances_ / sigma)


def HSIC(x: tf.Tensor, y: tf.Tensor, s_x: float, s_y: float):
    m = tf.shape(x)[0]  # batch size
    m_fl = tf.cast(m, tf.float32)
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = tf.eye(m) - 1.0 / m_fl * tf.ones((m, m))
    # H = H.float().cuda()
    HSIC = tf.linalg.trace(tf.matmul(L, tf.matmul(H, tf.matmul(K, H)))) / ((m_fl - 1.0) ** 2)
    return HSIC


def loss_fn(inputs, outputs, targets, name):
    """
    loss functions implementations based on the code from here:
    https://github.com/SJYuCNEL/Matrix-based-Dependence/blob/main/bike_sharing/loss.py

    HSIC, MI and MEE require the labels to be normalized in order to work (based on the original paper)

    Args:
        inputs:
        outputs:
        targets:
        name:

    Returns:

    """
    inputs_2d = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
    error = targets - outputs

    if name == 'mse':
        loss = keras.losses.mean_squared_error(targets, outputs)
    if name == 'HSIC':
        loss = HSIC(inputs_2d, error, s_x=2, s_y=1)
    if name == 'MI':
        loss = calculate_MI(inputs_2d, error, s_x=2, s_y=1)
    if name == 'MEE':
        loss = reyi_entropy(error, sigma=1)

    return loss