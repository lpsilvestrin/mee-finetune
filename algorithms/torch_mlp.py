import torch.nn as nn
from omegaconf import DictConfig
from torchvision.ops import MLP


class MyMLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, dropout: float, last_activation: str = 'linear'):
        super().__init__()
        ch = [in_channels] + hidden_channels

        activations = dict(
            linear=nn.Identity,
            softmax=nn.Softmax,
            relu=nn.ReLU
        )
        layers = []
        for i in range(1, len(ch)):
            layers.append(nn.Linear(ch[i-1], ch[i]))
            layers.append(activations['relu']())
            layers.append(nn.Dropout(p=dropout))

        fc = nn.Linear(ch[-1], out_channels)
        last_activation = activations[last_activation]()
        self.params = nn.Sequential(*layers, fc, last_activation)

    def forward(self, x):
        return self.params(x)


def build_mlp(nb_features: int, nb_out: int, config: DictConfig, last_activation: str = 'linear') -> MyMLP:
    if isinstance(config.hidden, DictConfig) or isinstance(config.hidden, dict):
        hidden = [config.hidden.units for _ in range(config.hidden.n)]
    else:
        hidden = config.hidden

    mlp = MyMLP(
        in_channels=nb_features,
        hidden_channels=hidden,
        out_channels=nb_out,
        dropout=config.dropout_rate,
        last_activation=last_activation
    )
    return mlp