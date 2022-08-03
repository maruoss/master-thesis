from datamodule_loop import Dataset, MyDataModule_Loop
from model.neuralnetwork import FFN

import pdb


def load_args(parser, mode: str, model: str):
    """load arguments for respective mode and model"""
    # pdb.set_trace()
    globals()[f"args_{model}"](parser)


def args_nn(parser_train):
    # EarlyStopping
    group = parser_train.add_argument_group("Early Stopping Configuration")
    # group.add_argument("--monitor", type=str, default="loss/val_loss")
    # group.add_argument("--es_mode", type=str, default="min")
    group.add_argument("--patience", type=int, default=5, #check check_val_every
                        help="number of bad validation epochs before stop, depends "
                        "on 'check_val_every' parameter as well")

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
    group.add_argument("--max_epochs", type=int, default=100)
    group.add_argument("--check_val_every", type=int, default=10)
    # parser = pl.Trainer.add_argparse_args(parser) # all the default trainer methods


def args_lin(parser_train):
    # Dataset args
    group = Dataset.add_model_specific_args(parser_train)

    # group = parser_train.add_argument_group("Model Configuration")
    group.add_argument("--loss", type=str, choices=["log_loss", "hinge"], 
                        required=True)
    group.add_argument("--pca", action="store_true")

    group = parser_train.add_argument_group("Sklearn Tune Configuration")
    group.add_argument("--max_iters", type=int, default=100)
    group.add_argument("--n_jobs", type=int, default=2) #how many trials in parallel


def args_svm(parser_train):
    # Dataset args
    group = Dataset.add_model_specific_args(parser_train)

    # group = parser_train.add_argument_group("Model Configuration")
    # group.add_argument("--loss", type=str, choices=["hinge", "log_loss"],
    #                     default="log_loss")
    group.add_argument("--pca", action="store_true")

    group = parser_train.add_argument_group("Sklearn Tune Configuration")
    group.add_argument("--max_iters", type=int, default=100)
    group.add_argument("--n_jobs", type=int, default=2) #how many trials in parallel


def args_rf(parser_train):
    # Dataset args
    group = Dataset.add_model_specific_args(parser_train)

    # group = parser_train.add_argument_group("Model Configuration")
    # group.add_argument("--loss", type=str, choices=["hinge", "log_loss"],
    #                     default="log_loss")
    group.add_argument("--pca", action="store_true")

    group = parser_train.add_argument_group("Sklearn Tune Configuration")
    group.add_argument("--n_estimators", type=int, default=1000)
    group.add_argument("--n_jobs", type=int, default=2) #how many trials in parallel


def args_xgb(parser_train):
    # Dataset args
    group = Dataset.add_model_specific_args(parser_train)

    # EarlyStopping
    group = parser_train.add_argument_group("Early Stopping Configuration")
    # group.add_argument("--monitor", type=str, default="loss/val_loss")
    # group.add_argument("--es_mode", type=str, default="min")
    group.add_argument("--patience", type=int, default=32, 
                            help="number of bad epochs before stop")

    group = parser_train.add_argument_group("Training Configuration")
    group.add_argument("--max_iters", type=int, default=1000)