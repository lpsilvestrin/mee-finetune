import numpy as np

import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from pytorch_lightning import LightningModule, seed_everything, Trainer

import wandb

from algorithms.torch_tcn import build_tcn


def train_custom_loss_tcn(train_x, train_y, test_sets, wandb_init):
    run = wandb.init(**wandb_init)
    config = wandb.config

    seed_everything(config.seed)

    rand_state = np.random.RandomState(config.seed)
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y,
                                                test_size=config.validation_split,
                                                random_state=rand_state)

    train_loader = DataLoader(TensorDataset(torch.tensor(tr_x).to(torch.float32), torch.tensor(tr_y).to(torch.float32)),
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_x).to(torch.float32), torch.tensor(val_y).to(torch.float32)),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4)

    out_shape = train_y.shape[1]

    in_shape = train_x.shape[1:]
    if config.model_type == 'tcn':
        in_features = in_shape[0]
        model = build_tcn(in_features, out_shape, config)
    # elif config.model_type == 'mlp':
    #     model = build_mlp(in_shape, out_shape, config)

    # model.double()
    if 'save_model' in config.keys():
        save_model = config.save_model
    else:
        save_model = False
    wandb_logger = WandbLogger(log_model=save_model)

    litmodel = My_LitModule(model, loss=config.loss_function, lr=config.learning_rate)
    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        accelerator='auto',
    )
    trainer.fit(litmodel, train_loader, val_loader)

    _, metrics = litmodel._get_preds_loss_metrics(val_loader.dataset.tensors)
    run.log({f"val/{k}": v for k, v in metrics.items()})

    for key, t_set in test_sets.items():
        tensor_t_set = torch.tensor(t_set[0], dtype=torch.float32), torch.tensor(t_set[1], dtype=torch.float32)
        _, metrics = litmodel._get_preds_loss_metrics(tensor_t_set)
        run.log({f"{key}/{k}": v for k, v in metrics.items()})

    wandb.finish()

    return model


class My_LitModule(LightningModule):

    def __init__(self, model, loss, lr=1e-3):
        '''method used to define our model parameters'''
        super().__init__()

        self.model = model

        self.loss = loss
        self.lr = lr
        self.metrics = dict(
            mse=mean_squared_error,
            mae=mean_absolute_error,
        )

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, metrics = self._get_preds_loss_metrics(batch)

        # Log loss and metric
        # self.log('train_loss', loss)
        for k, v in metrics.items():
            self.log(f'train_{k}', v)
        return metrics

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, metrics = self._get_preds_loss_metrics(batch)

        # Log loss and metric
        # self.log('train_loss', loss)
        for k, v in metrics.items():
            self.log(f'val_{k}', v)
        return metrics

    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_metrics(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        preds = self.model(x)
        loss = loss_fn(x, preds, y, self.loss)

        metrics = {k: m(preds, y) for k, m in self.metrics.items()}
        metrics['loss'] = loss
        return preds, metrics


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.001
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.001
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    normlize = Ixy / (torch.max(Hx, Hy) + 1e-16)
    return normlize


def GaussianKernelMatrix(x, sigma):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x, s_y):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduce=False)
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def loss_fn(inputs, outputs, targets, name):
    inputs_2d = inputs.reshape(inputs.shape[0], -1)
    error = targets - outputs
    # error = rmse(outputs, targets)
    if name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
    if name == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
    if name == 'rmse':
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(outputs, targets) + 1e-6)
    if name == 'MAE':
        criterion = torch.nn.L1Loss()
        loss = criterion(outputs, targets)

    if name == 'HSIC':
        loss = HSIC(inputs_2d, error, s_x=2, s_y=1)
    if name == 'MI':
        loss = calculate_MI(inputs_2d, error, s_x=2, s_y=1)
    if name == 'MEE':
        loss = reyi_entropy(error, sigma=1)
    if name == 'bias':
        loss = targets - outputs
        loss = torch.mean(loss, 0)
    return loss


