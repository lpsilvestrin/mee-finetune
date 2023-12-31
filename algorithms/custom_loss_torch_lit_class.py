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
        self.wandb_run = None
        self.debug = True
        # attribute for use in entropy loss function (MEE, MI, HSIC)
        self.sigma_y = 1
        self.sigma_x = 2
        # hacky attribute in order to get the output from trainer.test()
        # pylight issue documented here https://github.com/Lightning-AI/lightning/issues/1088
        self.test_output = None
        # store model bias during training for use later at test steps
        self.register_buffer("model_bias", torch.zeros(1))

    def forward(self, x):
        '''method used for inference input -> output'''
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, metrics = self._get_preds_loss_metrics(batch)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x) + self.model_bias
        loss = loss_fn(x, pred, y, self.loss, s_y=self.sigma_y, s_x=self.sigma_x, debug=self.debug)

        metrics = {k: m(pred, y) for k, m in self.metrics.items()}
        metrics['loss'] = loss
        res = y - pred
        return res, metrics

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, metrics = self._get_preds_loss_metrics(batch)

        # Log loss and metric
        metrics = {f'val_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        return metrics

    def test_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            # unzip residuals and metrics
            res_gath, metrics_gath = zip(*gathered)
            keys = metrics_gath[0].keys()
            metrics = {k: sum(output[k].mean() for output in metrics_gath) / len(metrics_gath) for k in keys}
            residuals = torch.concat(res_gath)
            self.test_output = (residuals, metrics)

    def training_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            # print(gathered)
            keys = gathered[0].keys()
            metrics = {f"train_{k}": sum(output[k].mean() for output in gathered) / len(outputs) for k in keys}
            self.wandb_run.log(metrics, step=self.current_epoch)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def validation_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            # print(gathered)
            keys = gathered[0].keys()
            metrics = {k: sum(output[k].mean() for output in gathered) / len(outputs) for k in keys}
            self.wandb_run.log(metrics, step=self.current_epoch)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

    def _get_preds_loss_metrics(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        preds = self(x)
        loss = loss_fn(x, preds, y, self.loss, s_y=self.sigma_y, s_x=self.sigma_x, debug=self.debug)

        metrics = {k: m(preds, y) for k, m in self.metrics.items()}
        metrics['loss'] = loss
        return preds, metrics


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(torch.square(x), -1).reshape((-1, 1))
    xxt = torch.mm(x, x.t())
    return -2 * xxt + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    median = torch.median(dist).item()
    print("median pw dist:", median)
    if sigma == 'median':
        sigma = median
    return torch.exp(-dist / sigma)


def print_off_diag_stats_and_pw_median(k, dist):
    n = k.shape[0]
    # select the off-diagonal elements (code from https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379)
    off_diag = k.flatten()[1:].view(n-1, n+1)[:, :-1]
    print("gram mat off-diag mean %.2f, std %.2f, median %.2f:" % (torch.mean(off_diag).item(), torch.std(off_diag).item(), torch.median(off_diag).item()))
    print("PW matrix median: %.2f" % torch.median(dist).item())


def renyi_entropy(x, sigma, debug=True):
    alpha = 1.001
    # k = calculate_gram_mat(x, sigma)
    dist = pairwise_distances(x)
    k = torch.exp(-dist / sigma)
    if debug:
        print_off_diag_stats_and_pw_median(k, dist)
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
    Hx = renyi_entropy(x, sigma=s_x)
    Hy = renyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    # normlize = Ixy / (torch.max(Hx, Hy) + 1e-16)
    # return normalize
    return Ixy


def GaussianKernelMatrix(x, sigma):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x, s_y, debug=True):
    m, _ = x.shape  # batch size
    distx = pairwise_distances(x)
    disty = pairwise_distances(y)
    K = torch.exp(-distx / s_x)
    L = torch.exp(-disty / s_y)
    if debug:
        print("stats X")
        print_off_diag_stats_and_pw_median(K, distx)
        print("stats Y")
        print_off_diag_stats_and_pw_median(L, disty)
    # K = GaussianKernelMatrix(x, s_x)
    # L = GaussianKernelMatrix(y, s_y)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    if x.device.type == 'cuda':
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


def loss_fn(inputs, outputs, targets, name, s_x=2, s_y=1, debug=True):
    inputs_2d = inputs.reshape(inputs.shape[0], -1)
    error = targets - outputs
    # print("input (min, max):", (torch.min(inputs).item(), torch.max(inputs).item()))
    # print("output (min, max):", (torch.min(outputs).item(), torch.max(outputs).item()))
    # print("label (min, max):", (torch.min(targets).item(), torch.max(targets).item()))
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
        loss = HSIC(inputs_2d, error, s_x=s_x, s_y=s_y, debug=debug)
    if name == 'MI':
        loss = calculate_MI(inputs_2d, error, s_x=s_x, s_y=s_y)
    if name == 'MEE':
        loss = renyi_entropy(error, sigma=s_y, debug=debug)
    if name == 'bias':
        loss = targets - outputs
        loss = torch.mean(loss, 0)
    return loss
