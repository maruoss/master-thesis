

from datamodule_loop import Dataset, MyDataModule_Loop
from model.neuralnetwork import FFN

import pdb

def load_args(parser, mode: str, model: str):
    """load arguments for respective mode and model"""
    # pdb.set_trace()
    globals()[f"args_{model}"](parser)
    if mode == "tune":
        globals()[f"args_tune"](parser)


def args_nn(parser_train):

    # Logger
    group = parser_train.add_argument_group("Logging Configuration")
    group.add_argument("--tag", type=str, default='')

    # EarlyStopping
    group = parser_train.add_argument_group("Early Stopping Configuration")
    # group.add_argument("--monitor", type=str, default="loss/val_loss")
    # group.add_argument("--es_mode", type=str, default="min")
    group.add_argument("--patience", type=int, default=3, help="number of bad epochs before stop")

    # ModelCheckpoint
    # group = parser.add_argument_group("Model Checkpoint Configuration")
    # group.add_argument("--monitor", type=str, default="loss/val_loss")
    # group.add_argument("--save_top_k", type=int, default=1)
    # group.add_argument("--check_mode", type=str, default="min")

    # dm
    # already implemented in model method: group = parser.add_argument_group("Data Module Configuration")
    group = MyDataModule_Loop.add_model_specific_args(parser_train)  #add additional arguments directly in class method

    # model
    # already implemented in model method: group = parser.add_argument_group("Model Configuration")
    group = FFN.add_model_specific_args(parser_train) #add additional arguments directly in class

    # trainer
    group = parser_train.add_argument_group("Training Configuration")
    group.add_argument("--max_epochs", type=int, default=2)
    group.add_argument("--check_val_every", type=int, default=1)
    # parser = pl.Trainer.add_argparse_args(parser) # all the default trainer methods

# additional arguments for tune, see load_args function
def args_tune(parser_tune):

    # Tune configuration
    group = parser_tune.add_argument_group("Tune Configuration")
    group.add_argument("--num_samples", type=int, default=1)
    group.add_argument("--gpus_per_trial", type=int, default=1)

    # ASHA scheduler configuration
    group = parser_tune.add_argument_group("ASHA Configuration")
    group.add_argument("--grace_period", type=int, default=1)
    group.add_argument("--reduction_factor", type=int, default=2)


def args_lin(parser_train):

    # Dataset args
    group = Dataset.add_model_specific_args(parser_train)

    # group = parser_train.add_argument_group("Model Configuration")
    group.add_argument("--loss", type=str, choices=["hinge", "log_loss"], 
                        required=True)
    group.add_argument("--pca", action="store_true")

    group = parser_train.add_argument_group("Sklearn Tune Configuration")
    group.add_argument("--max_iters", type=int, default=10)
    group.add_argument("--n_jobs", type=int, default=2)
    # group.add_argumet("--max_iter")

def args_svm(parser_train):

    # Dataset args
    group = Dataset.add_model_specific_args(parser_train)

    # group = parser_train.add_argument_group("Model Configuration")
    # group.add_argument("--loss", type=str, choices=["hinge", "log_loss"],
    #                     default="log_loss")
    group.add_argument("--pca", action="store_true")

    group = parser_train.add_argument_group("Sklearn Tune Configuration")
    group.add_argument("--max_iters", type=int, default=10)
    group.add_argument("--n_jobs", type=int, default=2)
    # group.add_argumet("--max_iter")