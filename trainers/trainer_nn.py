
from pathlib import Path
import pdb
import numpy as np
import pandas as pd
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from yaml import serialize

from datamodule_loop import MyDataModule_Loop
from utils.logger import create_foldername, serialize_args, serialize_config, params_to_dict
from model.neuralnetwork import FFN

from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from torch import nn

def nn_train(args, year_idx, time, ckpt_path=None, config=None):
    dm = MyDataModule_Loop(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        batch_size=args.batch_size,
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        label_fn=args.label_fn
    )
    dm.setup() #needed for model parameters
    print("dm is set up!")
    model = FFN(
        input_dim=dm.input_dim,
        num_classes=dm.num_classes,
        class_weights=dm.class_weights,
        no_class_weights=args.no_class_weights,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_hidden=args.n_hidden,
        batch_norm=args.no_batch_norm,
        dropout=args.no_dropout,
        drop_prob=args.drop_prob,
    )
    print("Model is loaded!")

    # specify which parameters will be added/ removed from logging folder name
    to_add = {"max_epochs": args.max_epochs}
    # to_exclude = ["class_weights", "year_idx", "config"]
    to_exclude = []
    # to_exclude = ["path", "dm"] -> moved to parameter "ignore" of save_hyperparameters

    # Set logging directory
    log_dir = "logs/train/nn_loops"
    # name = create_foldername(model=model, dm=dm, to_add=to_add, to_exclude=to_exclude, tag=args.tag)
    name = time
    train_year_end = int(dm.dates.iloc[:dm.eoy_train].iloc[-1].strftime("%Y"))
    val_year_end = train_year_end+args.val_length
    version = f"train{train_year_end}_val{val_year_end}"

    # save hyperparams as .json
    summary_path = Path.cwd()/log_dir/time
    summary_path.mkdir(exist_ok=True, parents=True)
    with open(summary_path/"params.json", 'w') as f:
        json.dump(params_to_dict(model=model, dm=dm, to_add=to_add, 
                    to_exclude=to_exclude, tag=args.tag), fp=f, indent=3)
    
    # save args to json
    args_dict = serialize_args(args.__dict__) #functions are not serializable
    with open(summary_path/'args.json', 'w') as f:
        json.dump(args_dict, f, indent=3)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=name,
        version=version,
    )

    early_stop_callback = EarlyStopping(
        monitor="loss/val_loss", 
        mode="min", 
        patience=args.patience,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_loss",
        save_top_k=1,
        mode="min",
        filename='epoch={epoch}-val_loss={loss/val_loss:.3f}-val_bacc={bal_acc/val:.4f}',
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
    if args.refit and ckpt_path is not None:
        print(f"Loading model from path at {ckpt_path}")
        # class weights from this loop have to be used
        model = FFN.load_from_checkpoint(ckpt_path, class_weights=dm.class_weights)
    trainer.fit(model=model, datamodule=dm)

    print("Save validation metrics of best model...")
    # will log to tensorboard of trainer.fit results -> last iteration
    val_results = trainer.validate(ckpt_path="best", datamodule=dm)[0] #is a list
    # new ckpt path
    if args.refit:
        ckpt_path = checkpoint_callback.best_model_path

    if not args.no_predict:
        print("Predict on test set and save to csv...")
        preds = trainer.predict(ckpt_path="best", datamodule=dm)
        preds_argmax = preds[0].argmax(dim=1).numpy() # assumes batchsize is whole testset
        preds_argmax_df = pd.DataFrame(preds_argmax, columns=["pred"])
        test_year_end = val_year_end + args.test_length
        # prediction path
        save_to_dir = Path(Path.cwd(),log_dir, name, version, f"prediction{test_year_end}.csv")
        preds_argmax_df.to_csv(save_to_dir, index_label="id")

    # val summary path
    summary_path = Path(Path.cwd(), log_dir, name)

    # Rename metric names to align with tune
    val_results["val_acc"] = val_results.pop("acc/val")
    val_results["val_bal_acc"] = val_results.pop("bal_acc/val")
    val_results["val_loss"] = val_results.pop("loss/val_loss")

    return val_results, summary_path, ckpt_path, config # dictionary of metrics, , path to save summary


# ***********************************************************************************

def inner_nn_tune(config, args, year_idx, ckpt_path):
    # needed for reproducibility, will seed trainer (init of weights in NN?)
    pl.seed_everything(args.seed, workers=True)

    dm = MyDataModule_Loop(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        batch_size=config["batch_size"],
        init_train_length=args.init_train_length,
        val_length=args.val_length,
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

    if ckpt_path:
        # pdb.set_trace()
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
        "train_acc" : "acc/train",
        "train_bal_acc": "bal_acc/train",
        "val_loss": "loss/val_loss",
        "val_acc": "acc/val",
        "val_bal_acc": "bal_acc/val",
        "mean_pred": "mean_pred",
    },
    filename="checkpoint",
    on="validation_end")
    
    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=args.max_epochs,
        gpus=args.gpus_per_trial,
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
        "lr": tune.qloguniform(1e-4, 1e-1, 5e-5), #round to 5e-5 steps
        "batch_size": tune.choice([256, 512]),
        # "lr": 1e-6, #round to 5e-5 steps
        # "batch_size": 128,
        # ***
        "hidden_dim": tune.choice([50, 100]),
        "n_hidden" : tune.choice([1, 2, 3]),
        "batch_norm" : tune.choice([True, False]),
        "dropout" : tune.choice([True, False]),
        "drop_prob" : tune.quniform(0, 0.5, 0.1), #round to 0.1 steps
#         "hidden_dim": tune.choice([32]),
#         "lr": tune.choice([1e-2]),
#         "batch_size": tune.choice([512]),
    }

    if args.refit and start_config is not None:
        pdb.set_trace()
        config = start_config.copy()
        config.update({
            "lr": tune.qloguniform(1e-4, 1e-1, 5e-5), #round to 5e-5 steps
            "batch_size": tune.choice([256, 512]),
            # "lr": 1e-6, #round to 5e-5 steps
            # "batch_size": 128,
        })

    best_result, summary_path, best_ckpt_path, best_config = \
    nn_tune_from_config(args, year_idx, time, ckpt_path, config=config)

    return best_result, summary_path, best_ckpt_path, best_config


def nn_tune_from_config(args, year_idx, time, ckpt_path, config: dict):

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=args.grace_period,
        reduction_factor=args.reduction_factor)

    reporter = CLIReporter(
        parameter_columns=["hidden_dim", "lr", "batch_size"],
        metric_columns=["train_loss", "val_loss", "val_bal_acc", "mean_pred", 
        "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(inner_nn_tune,
                                                    args=args,
                                                    year_idx=year_idx,
                                                    ckpt_path=ckpt_path,
                                                   )
    resources_per_trial = {"cpu": 1, "gpu": args.gpus_per_trial}

    # Set logging directory for tune.run
    log_dir = "./logs/tune/nn_loops"
    # name = time+"_"+string_from_config(config) # config into path 
    # CAREFUL: will give error if directory path is too large
    train_year_end = 1996 + args.init_train_length + year_idx - 1
    val_year_end = train_year_end + args.val_length
    years = f"train{train_year_end}_val{val_year_end}"
    name = time+"\\"+years

    # save config space as .json
    summary_path = Path.cwd()/log_dir/time
    summary_path.mkdir(exist_ok=True, parents=True)
    with open(summary_path/"config.json", 'w') as f:
        json.dump(serialize_config(config), fp=f, indent=3)

    # save args to json
    args_dict = serialize_args(args.__dict__) #functions are not serializable
    with open(summary_path/'args.json', 'w') as f:
        json.dump(args_dict, f, indent=3)


    analysis = tune.run(train_fn_with_parameters,
        local_dir=log_dir,
        resources_per_trial=resources_per_trial,
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=name,
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
    

    best_result_per_trial_df = analysis.dataframe(metric="val_loss", 
                                            mode="min").sort_values("val_loss")
    # save df to folder?
    best_result = best_result_per_trial_df.iloc[0, :].to_dict() #take best values of best trial

    # test prediction: best checkpoint out of all trials
    best_trial = analysis.get_best_trial("val_loss", "min", scope="all")
    if args.refit:
        config = best_trial.config
        ckpt_path = Path(analysis.get_best_checkpoint(best_trial).get_internal_representation()[1], "checkpoint")
    
    if not args.no_predict:
        best_path = Path(analysis.get_best_checkpoint(best_trial).get_internal_representation()[1], "checkpoint")
        print(f"Loading model from path at {ckpt_path}")
        model = FFN.load_from_checkpoint(best_path)
        dm = MyDataModule_Loop(
            path=args.path_data,
            year_idx=year_idx,
            dataset=args.dataset,
            batch_size=1, #any number, doesnt matter for predict
            init_train_length=args.init_train_length,
            val_length=args.val_length,
            label_fn=args.label_fn,
            # config=model.hparams.config, # so that config is not hyperparam search again
        )
        trainer = pl.Trainer(
            deterministic=True,
            gpus=args.gpus_per_trial,
        )
        preds = trainer.predict(model=model, datamodule=dm)
        preds_argmax = preds[0].argmax(dim=1).numpy() # assumes batchsize is whole testset
        preds_argmax_df = pd.DataFrame(preds_argmax, columns=["pred"])
        test_year_end = val_year_end + args.test_length
        # prediction path
        save_to_dir = Path(Path.cwd(),log_dir, name, f"prediction{test_year_end}.csv")
        preds_argmax_df.to_csv(save_to_dir, index_label="id")
        
    return best_result, summary_path, ckpt_path, config 
    #dictionary of best metrics and config, path
