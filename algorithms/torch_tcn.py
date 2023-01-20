import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    base TCN implementation from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, num_inputs: int, channels: list, num_blocks: int, num_out: int, kernel_size: int=2, dropout: float =0.2):
        super(TCN, self).__init__()
        # self.save_hyperparameters()
        tcn_list = [TemporalConvNet(num_inputs, channels, kernel_size, dropout)]
        for _ in range(num_blocks-1):
            tcn_list.append(TemporalConvNet(channels[-1], channels, kernel_size, dropout))

        self.tcn = nn.Sequential(*tcn_list)
        self.fc = nn.Linear(channels[-1], num_out)

    def forward(self, x):
        conv_out = self.tcn(x)
        return self.fc(conv_out[:, :, -1].squeeze(-1))


def build_tcn(
        nb_features: int,
        nb_out: int,
        config: DictConfig) -> TCN:
    num_blocks = 1
    if config.tcn2 is True:
        num_blocks = 2
    if 'tcn' in config:
        filters = [config.tcn['filters'] for _ in range(config.tcn['dilations'])]
    else:
        filters = config.filters

    tcn = TCN(
        num_inputs=nb_features,
        channels=filters,
        num_blocks=num_blocks,
        num_out=nb_out,
        kernel_size=config.kernel_size,
        dropout=config.dropout_rate
    )
    return tcn

