import numpy as np

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from pytorch_lightning import LightningModule, seed_everything, Trainer

import wandb

from algorithms.custom_loss_torch_lit_class import PyLitModelWrapper
from algorithms.torch_mlp import build_mlp
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
    in_features = in_shape[0]
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

    ckp_callback = ModelCheckpoint(dirpath='pylit_chkpt/', monitor='val_loss', mode='min', filename=run.id)
    earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=config.early_stop_patience)

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

    _, metrics = litmodel._get_preds_loss_metrics(val_loader.dataset.tensors)
    run.log({f"val/{k}": v for k, v in metrics.items()})

    for key, t_set in test_sets.items():
        tensor_t_set = torch.tensor(t_set[0], dtype=torch.float32), torch.tensor(t_set[1], dtype=torch.float32)
        _, metrics = litmodel._get_preds_loss_metrics(tensor_t_set)
        run.log({f"{key}/{k}": v for k, v in metrics.items()})

    wandb.finish()

    return litmodel




