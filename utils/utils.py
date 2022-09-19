import keras
from keras import Model, Input
from omegaconf import DictConfig
from keras.layers import Dense, Dropout


def evaluate(x, y, model, metrics):
    pred = model.predict(x)
    result = []
    for met in metrics:
        met.update_state(y, pred)
        result.append(met.result().numpy())
        met.reset_state()
    return result


def build_mlp(nb_features: int, nb_out: int, config: DictConfig, last_activation: str = 'linear') -> Model:
    if isinstance(config.hidden, DictConfig) or isinstance(config.hidden, dict):
        hidden = [config.hidden.units for _ in range(config.hidden.n)]
    else:
        hidden = config.hidden

    i = Input(shape=(nb_features))

    l2 = config.l2_reg if 'l2_reg' in config else 0
    l2_reg = keras.regularizers.L2(l2)

    m = i
    for n in hidden:
        m = Dense(n,
                  activation='relu',
                  kernel_regularizer=l2_reg)(m)
        Dropout(config.dropout_rate)(m)

    m = Dense(nb_out,
              activation=last_activation,
              kernel_regularizer=l2_reg)(m)
    return Model(inputs=[i], outputs=[m])