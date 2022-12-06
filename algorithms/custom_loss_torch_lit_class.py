import torch
from pytorch_lightning import LightningModule
from torch.optim.adam import Adam
import torch.nn as nn


class PyLitModelWrapper(LightningModule):

    def __init__(self, model, loss, metrics, lr=1e-3, l2_reg=.0):
    # def __init__(self, loss, lr, metrics={}):
        '''method used to define our model parameters'''
        super().__init__()

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters(ignore='model')

        self.model = model

        self.loss = loss
        self.lr = lr
        self.metrics = metrics
        self.l2_reg = l2_reg

    def forward(self, x):
        '''method used for inference input -> output'''

        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, metrics = self._get_preds_loss_metrics(batch)

        # Log loss and metric
        # self.log('train_loss', loss)
        for k, v in metrics.items():
            self.log(f'train_{k}', v, on_step=False, on_epoch=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, metrics = self._get_preds_loss_metrics(batch)

        # Log loss and metric
        # self.log('train_loss', loss)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

    def _get_preds_loss_metrics(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        preds = self(x)
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
    # normlize = Ixy / (torch.max(Hx, Hy) + 1e-16)
    # return normalize
    return Ixy


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
