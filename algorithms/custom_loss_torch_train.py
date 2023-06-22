import os

import numpy as np

import torch

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from pytorch_lightning import LightningModule, seed_everything, Trainer

import wandb

from algorithms.custom_loss_torch_lit_class import PyLitModelWrapper
from algorithms.torch_mlp import build_mlp
from algorithms.torch_tcn import build_tcn


def get_gaussian_kernel_size(dataset, loss_function):
    # set optional loss parameters for MEE, MI, HSIC
    gaussian_kernel_dict = {
        'src': dict(y=0.7, x=400),
        'tar1': dict(y=0.5, x=800),
        'tar2': dict(y=1, x=450),
        'tar3': dict(y=0.5, x=800),
        'bpm10_src': dict(y=0.5, x=300),
        'bpm10_tar': dict(y=0.3, x=300),
        "bike11_src": dict(y=0.3, x=250),
        "bike11_tar": dict(y=0.8, x=200)
    }
    return gaussian_kernel_dict[dataset]


def train_torch(train_x, train_y, test_sets, wandb_init, model=None):
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
    in_features = in_shape[0]
    if model is None:
        if config.model_type == 'tcn':
            # model = build_tcn(in_features, out_shape, config)
            model = build_tcn(in_features, out_shape, config)
        elif config.model_type == 'mlp':
            model = build_mlp(in_features, out_shape, config)

    # model.double()
    if 'save_model' in config.keys():
        save_model = config.save_model
    else:
        save_model = False
    wandb_logger = WandbLogger(log_model=save_model)

    early_stop_criteria = 'val_loss' if 'early_stop_criteria' not in config else config.early_stop_criteria
    ckp_callback = ModelCheckpoint(dirpath='pylit_chkpt/', monitor=early_stop_criteria, mode='min', filename=run.id)
    earlystop_callback = EarlyStopping(monitor=early_stop_criteria, mode='min', patience=config.early_stop_patience)

    metrics = dict(
        mse=mean_squared_error,
        mae=mean_absolute_error,
    )
    litmodel = PyLitModelWrapper(
        model,
        metrics=metrics,
        loss=config.loss_function,
        lr=config.learning_rate,
        l2_reg=config.l2_reg
    )
    # assign the wandb run object to log training statistics
    litmodel.wandb_run = run
    gks = get_gaussian_kernel_size(config.train_dataset, config.loss_function)
    litmodel.sigma_x = gks['x']
    litmodel.sigma_y = gks['y']
    if 'sigma_y' in config:
        litmodel.sigma_y = config.sigma_y
    if 'sigma_x' in config:
        litmodel.sigma_x = config.sigma_x

    debug_mode = False
    if 'debug_mode' in config:
        debug_mode = config.debug_mode

    if debug_mode:
        wandb_logger.watch(litmodel, log='all', log_graph=False, log_freq=10)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        accelerator='auto',
        callbacks=[ckp_callback, earlystop_callback],
    )
    trainer.fit(litmodel, train_loader, val_loader)

    # load best model
    litmodel = PyLitModelWrapper.load_from_checkpoint(ckp_callback.best_model_path, model=model)
    # eval mode: disable randomness, dropout, etc before running tests
    litmodel.eval()

    if config.loss_function in ["MEE", "MI", "HSIC"]:
        print("computing model bias --------------")
        trainer.test(litmodel, train_loader)
        res, _ = litmodel.test_output
        litmodel.model_bias = res.mean()

    trainer.test(litmodel, val_loader)
    _, metrics = litmodel.test_output
    run.log({f"val/{k}": v for k, v in metrics.items()})

    for key, t_set in test_sets.items():
        test_loader = DataLoader(
            TensorDataset(torch.tensor(t_set[0], dtype=torch.float32), torch.tensor(t_set[1], dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4)

        trainer.test(litmodel, test_loader)
        res, metrics = litmodel.test_output
        run.log({f"{key}/{k}": v for k, v in metrics.items()})
        res_filename = os.path.join(run.dir, f"{key}_test_residuals.npy")
        np.save(res_filename, res.cpu().numpy())
        run.save(res_filename)

    wandb.finish()

    return litmodel, ckp_callback.best_model_path
