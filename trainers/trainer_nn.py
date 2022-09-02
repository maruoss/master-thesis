import math
from pathlib import Path
import pdb
import numpy as np
import pandas as pd
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil
import torch

from datamodule import DataModule
from utils.helper import set_tune_log_dir
from utils.logger import serialize_config
from model.neuralnetwork import FFN

from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from torch import nn


def inner_nn_tune(config, args, year_idx, ckpt_path):
    # Needed for reproducibility, will seed trainer (init of weights in NN, ...).
    pl.seed_everything(args.seed, workers=True)

    dm = DataModule(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        batch_size=config["batch_size"],
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        test_length=args.test_length,
        label_fn=args.label_fn,
    )
    dm.setup()

    model = FFN(
        input_dim=dm.input_dim,
        num_classes=dm.num_classes,
        class_weights=dm.class_weights,
        no_class_weights=False,
        hidden_dim=config["hidden_dim"],
        learning_rate=config["lr"],
        n_hidden=config["n_hidden"],
        batch_norm=config["batch_norm"],
        dropout=config["dropout"],
        drop_prob=config["drop_prob"],
    )

    # If --refit is active, ckpt_path will be best model of last loop.
    if ckpt_path:
        print(f"Loading model from path at {ckpt_path}")
        model = FFN.load_from_checkpoint(ckpt_path)

    early_stop_callback = EarlyStopping(
        monitor="loss/val_loss", 
        mode="min", 
        patience=args.patience,
    )

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="loss/val_loss",
    #     save_top_k=1,
    #     mode="min",
    #     filename='epoch={epoch}-val_loss={loss/val_loss:.3f}-val_bacc={bal_acc/val:.4f}',
    #     auto_insert_metric_name=False,
    # )

    tune_callback = TuneReportCheckpointCallback(
        metrics={
            "train_loss": "loss/loss",
            "train_acc" : "acc/train", # currently cant be found by raytune
            "train_bal_acc": "bal_acc/train", # currenctly cant be found by raytune
            "val_loss": "loss/val_loss",
            "val_acc": "acc/val",
            "val_bal_acc": "bal_acc/val",
            "mean_pred": "mean_pred",
        },
        filename="checkpoint",
        on="validation_end"
    )
    
    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=args.max_epochs,
        gpus=math.ceil(args.gpus_per_trial), #fractional gpus here not possible
        logger=pl.loggers.TensorBoardLogger(
        save_dir=tune.get_trial_dir(), name="", version="."),
        enable_progress_bar=True,
        check_val_every_n_epoch=args.check_val_every,
        callbacks=[ 
                #    checkpoint_callback, 
                   early_stop_callback,
                   tune_callback, 
                  ],
        num_sanity_val_steps=2,
        enable_checkpointing=False, #disable default "last model" checkpointer
    )

    trainer.fit(model, datamodule=dm)
    


def nn_tune(args, year_idx, time, ckpt_path, start_config: dict):

    config = {
        "lr": tune.qloguniform(1e-6, 1e-1, 5e-7), #round to 5e-7 steps
        "batch_size": tune.choice([128, 256, 512]),
        # "lr": 1e-6, #round to 5e-5 steps
        # "batch_size": 128,
        # ***
        "hidden_dim": tune.choice([25, 50, 100]),
        "n_hidden" : tune.choice([1, 2, 3]),
        "batch_norm" : tune.choice([True, False]),
        "dropout" : tune.choice([True, False]),
        "drop_prob" : tune.quniform(0, 0.5, 0.05), #round to 0.1 steps
#         "hidden_dim": tune.choice([32]),
#         "lr": tune.choice([1e-2]),
#         "batch_size": tune.choice([512]),
    }

    if args.refit and start_config is not None:
        config = start_config.copy()
        config.update({
            "lr": tune.qloguniform(1e-6, 1e-1, 5e-7), #round to 5e-7 steps
            "batch_size": tune.choice([128, 256, 512]),
            # "lr": 1e-6, #round to 5e-5 steps
            # "batch_size": 128,
        })

    best_result, summary_path, best_ckpt_path, best_config = \
    nn_tune_from_config(args, year_idx, time, ckpt_path, config=config)

    # dict, Path obj., Path obj, dict
    return best_result, summary_path, best_ckpt_path, best_config


def nn_tune_from_config(args, year_idx, time, ckpt_path, config: dict):

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=args.grace_pct*args.max_epochs, #how many epochs for sure.
        reduction_factor=args.reduction_factor
    )
    reporter = CLIReporter(
        parameter_columns=["hidden_dim", "lr", "batch_size"],
        metric_columns=["train_loss", "val_loss", "val_bal_acc", "mean_pred", 
                        "training_iteration"],
        max_report_frequency=120, 
        print_intermediate_tables=True
    )
    train_fn_with_parameters = tune.with_parameters(inner_nn_tune,
                                                    args=args,
                                                    year_idx=year_idx,
                                                    ckpt_path=ckpt_path,
                                                   )
    # Assuming machine has 8 cores per 1 gpu
    resources_per_trial = {
                        "cpu": args.gpus_per_trial * 8,
                        "gpu": args.gpus_per_trial
                        }

    val_year_end, loop_foldername, exp_path = set_tune_log_dir(args, year_idx, time, config)

    analysis = tune.run(
        train_fn_with_parameters,
        local_dir=exp_path,
        resources_per_trial=resources_per_trial,
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=loop_foldername,
        fail_fast=True, # stop all trials as soon as any trial errors
        keep_checkpoints_num=1, # only keep best checkpoint
        checkpoint_score_attr="min-val_loss",
        )

    print("Best hyperparameters found were: ", analysis.best_config)

    best_last_trial = analysis.get_best_trial("val_loss", "min", "last") #change "last" to "all" for global min
    print("Best trial among last epoch config: {}".format(best_last_trial.config))
    print("Best trial >>last epoch<< validation loss: {}".format(
        best_last_trial.last_result["val_loss"]))
    print("Best trial >>last epoch<< validation balanced accuracy: {}".format(
        best_last_trial.last_result["val_bal_acc"]))
    
    # Loop Path for best_config and prediction.csv.
    loop_path = Path(Path.cwd(), exp_path, loop_foldername)
    
    # should yield the data for the best checkpoint, if checkpoints are made every epoch
    best_result_per_trial_df = analysis.dataframe(metric="val_loss", 
                                                  mode="min").sort_values("val_loss")
    # save df to folder?
    best_result = best_result_per_trial_df.iloc[0, :].to_dict() #take best values of best trial
    pd.DataFrame([best_result]).T.to_csv(loop_path/"best_result.csv") #save best results to csv.

    # test prediction: use best checkpoint out of all trials
    best_trial = analysis.get_best_trial("val_loss", "min", scope="all")
    best_config = best_trial.config

    # Save best config as .json.
    with open(loop_path/"best_config.json", 'w') as f:
        json.dump(serialize_config(best_config), fp=f, indent=3)

    if args.refit:
        config = best_trial.config
        ckpt_path = Path(analysis.get_best_checkpoint(best_trial)
                        .get_internal_representation()[1], "checkpoint")
    
    if not args.no_predict:
        # load best model
        best_path = Path(analysis.get_best_checkpoint(best_trial).get_internal_representation()[1],
                        "checkpoint")
        # Copy best model checkpoint to loop folder for later analysis.
        test_year_end = val_year_end + args.test_length
        shutil.copy2(best_path, loop_path/f"best_ckpt{test_year_end}")
        print(f"Loading model to predict from path: {best_path}")
        model = FFN.load_from_checkpoint(best_path)
        dm = DataModule(
            path=args.path_data,
            year_idx=year_idx,
            dataset=args.dataset,
            batch_size=best_config["batch_size"], #take batchsize of best config.
            init_train_length=args.init_train_length,
            val_length=args.val_length,
            test_length=args.test_length,
            label_fn=args.label_fn,
            # config=model.hparams.config, # so that config is not hyperparam search again
        )
        trainer = pl.Trainer(
            deterministic=True,
            gpus=math.ceil(args.gpus_per_trial), #fractional gpus here not possible.
            logger=False, #deactivate logging for prediction
        )
        # predict
        preds = trainer.predict(model=model, datamodule=dm) #returns list of batch predictions.
        preds = torch.cat(preds) #preds is a list already of [batch_size, num_classes]. 
        preds_argmax = preds.argmax(dim=1).numpy()
        preds_argmax_df = pd.DataFrame(preds_argmax, columns=["pred"])
        # prediction path
        save_to_dir = loop_path/f"prediction{test_year_end}.csv"
        preds_argmax_df.to_csv(save_to_dir, index_label="id")
        
    # dict, Path obj., Path obj, dict
    return best_result, exp_path, ckpt_path, config 
