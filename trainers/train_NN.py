
import pdb
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datamodule_loop import MyDataModule_Loop
from utils.logger import create_foldername, string_from_config
from model.neuralnetwork import FFN

from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

def nn_loop(args, year_idx, time):
    dm = MyDataModule_Loop(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        batch_size=args.batch_size,
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        # start_val=args.start_val,
        # start_test=args.start_test,
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
    to_exclude = ["class_weights", "year_idx"]
    # to_exclude = ["path", "dm"] -> moved to parameter "ignore" of save_hyperparameters

    # Set logging directory
    log_dir = "logs/train_loops"
    name = create_foldername(model=model, dm=dm, to_add=to_add, to_exclude=to_exclude, tag=args.tag)
    name = name+"/"+time
    train_year_end = 1996+args.init_train_length+year_idx
    val_year_end = 1996+args.init_train_length+year_idx+args.val_length
    version = f"train{train_year_end}_val{val_year_end}"

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
        filename='epoch={epoch}-val_loss={loss/val_loss:.3f}-val_bacc={bal_accuracy/val:.4f}',
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


# used in main_loop, function needs to be defined before its usage
def nn_tune(args, year_idx, time):
        tune_nn_asha(args, year_idx, time)

def train_nn_tune(config, args, year_idx, checkpoint_dir=None):
#     data_dir = os.path.expanduser(data_dir)

    dm = MyDataModule_Loop(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        batch_size=None,
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        # start_val=args.start_val,
        # start_test=args.start_test,
        label_fn=args.label_fn,
        config=config,
    )
    
    dm.setup()
    
    model = FFN(
    input_dim=dm.input_dim,
    num_classes=dm.num_classes,
    class_weights=dm.class_weights,
    no_class_weights=False,
    hidden_dim=None,
    learning_rate=None,
    config=config,
    )

    early_stop_callback = EarlyStopping(
        monitor="loss/val_loss", 
        mode="min", 
        patience=args.patience,
    )

    tune_callback = TuneReportCheckpointCallback(
    metrics={
        "val_loss": "loss/val_loss",
        "val_bal_acc": "bal_acc/val"
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
        callbacks=[tune_callback, 
#                    checkpoint_callback, 
                   early_stop_callback
                  ],
        num_sanity_val_steps=2,
    )

    trainer.fit(model, datamodule=dm)

def tune_nn_asha(args, year_idx, time):

    config = {
        "hidden_dim": tune.choice([50, 100]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([256, 512]),
#         "hidden_dim": tune.choice([32]),
#         "lr": tune.choice([1e-2]),
#         "batch_size": tune.choice([512]),
    }

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=args.grace_period,
        reduction_factor=args.reduction_factor)

    reporter = CLIReporter(
        parameter_columns=["hidden_dim", "lr", "batch_size"],
        metric_columns=["val_loss", "val_bal_acc" ,"training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_nn_tune,
                                                    args=args,
                                                    year_idx=year_idx,
#                                                     data_dir=data_dir,
                                                   )
    resources_per_trial = {"cpu": 1, "gpu": args.gpus_per_trial}

    # Set logging directory for tune.run
    log_dir = "./logs/tune_loops"
    # name = create_foldername(model=model, dm=dm, to_add=to_add, to_exclude=to_exclude, tag=args.tag)
    name = time+"_"+string_from_config(config)
    train_year_end = 1996+args.init_train_length+year_idx
    val_year_end = 1996+args.init_train_length+year_idx+args.val_length
    years = f"train{train_year_end}_val{val_year_end}"
    name = name+"\\"+years
    # pdb.set_trace()

    train_year_end = 1996+args.init_train_length+year_idx
    val_year_end = 1996+args.init_train_length+year_idx+args.val_length
    years = f"train{train_year_end}_val{val_year_end}"
    
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
        fail_fast=True #stop all trials as soon as any trial errors
        )

    print("Best hyperparameters found were: ", analysis.best_config)
    
    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial >>last<< validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial >>last epoch<< validation accuracy: {}".format(
        best_trial.last_result["val_bal_acc"]))
    
    return analysis
