import pdb
import numpy as np
import pandas as pd
import pathlib
import time
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch import nn
from pytorch_lightning import seed_everything
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
import torchmetrics

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from argparse import ArgumentParser

from datamodule import MyDataModule
from model.neuralnetwork import FFN
from utils.logger import log_foldername


def train(args):
    
    dm = MyDataModule(
        path=path_data,
        dataset=args.dataset,
        batch_size=args.batch_size,
        start_val=args.start_val,
        start_test=args.start_test,
        label_fn=args.label_fn
    )
    dm.setup() #needed for model parameters
    print("dm is set up!")
    model = FFN(
        # dm=dm,
        input_dim=dm.input_dim,
        num_classes=dm.num_classes,
        class_weights=dm.class_weights,
        no_class_weights=args.no_class_weights,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )
    print("Model is loaded!")

    # specify which parameters will be added/ removed from logging folder name
    to_add = {"max_epochs": args.max_epochs}
    to_exclude = []
    # to_exclude = ["path", "dm"] -> moved to parameter "ignore" of save_hyperparameters

    # Set logging directory
    log_dir = "logs"
    name = log_foldername(model=model, dm=dm, to_add=to_add, to_exclude=to_exclude, tag=args.tag)
    version = datetime.now().strftime("%Y%m%d%H%M%S")

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=name,
        version=version,
    )

    early_stop_callback = EarlyStopping(
        monitor="loss/val_loss", 
        mode="min", 
        patience=args.patience
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_loss",
        save_top_k=1,
        mode="min",
        filename='epoch={epoch}-val_loss={loss/val_loss:.3f}-\
            val_bacc={bal_accuracy/val:.4f}',
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        deterministic=True,
        gpus=1,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=2,
    )
    print("Fitting the model...")
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    seed_everything(42, workers=True)

    # Set path
    path_data = pathlib.Path(
        r"C:\Users\Mathiass\OneDrive - Universität Zürich UZH\Documents\mt_literature\data"
    )
    
    parser = ArgumentParser(description="Master Thesis Mathias Ruoss - Option\
        Return Classification")
    # Logger
    logging = parser.add_argument_group("Logging Configuration")
    logging.add_argument("--tag", type=str, default='')

    # EarlyStopping
    earlystop = parser.add_argument_group("Early Stopping Configuration")
    # earlystop.add_argument("--monitor", type=str, default="loss/val_loss")
    # earlystop.add_argument("--es_mode", type=str, default="min")
    earlystop.add_argument("--patience", type=int, default=3)

    # ModelCheckpoint
    # modelcheck = parser.add_argument_group("Model Checkpoint Configuration")
    # modelcheck.add_argument("--monitor", type=str, default="loss/val_loss")
    # modelcheck.add_argument("--save_top_k", type=int, default=1)
    # modelcheck.add_argument("--check_mode", type=str, default="min")

    # dm
    datamodule = parser.add_argument_group("Data Module Configuration")
    datamodule = MyDataModule.add_model_specific_args(datamodule)  #add additional arguments directly in class method

    # model
    model = parser.add_argument_group("Model Configuration")
    model = FFN.add_model_specific_args(model) #add additional arguments directly in class

    # trainer
    trainer = parser.add_argument_group("Trainer Configuration")
    trainer.add_argument("--max_epochs", type=int, default=20)
    trainer.add_argument("--check_val_every", type=int, default=1)
    # parser = pl.Trainer.add_argparse_args(parser) # all the default trainer methods

    args = parser.parse_args([])

    train(args)
