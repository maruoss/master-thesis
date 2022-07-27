from pathlib import Path
import pdb
from datetime import datetime
import pandas as pd

from pytorch_lightning import seed_everything
from argparse import ArgumentParser
from trainers.trainer_xgb import xgb_tune
from utils.helper import summary_to_csv
from trainers.trainer_sk import sk_run
from utils.arguments import load_args
from trainers.trainer_nn import nn_train, nn_tune


def run(args, year_idx, time, ckpt_path, config):
    """Run the specified model and return dictionaries and path for next loop."""
    fun_dict = {
                "nn_train": nn_train, 
                "nn_tune": nn_tune, 
                "lin_tune": sk_run,
                "svm_tune": sk_run,
                "xgb_tune": xgb_tune,
                }

    # no checkpoint here -> no refit possible
    best_result, summary_path, ckpt_path, config \
        = fun_dict[f"{args.model}_{args.mode}"](args, year_idx, time, ckpt_path, config)

    return best_result, summary_path, ckpt_path, config

def looper(args):
    """Loop over all train, val splits: 1996 - 2021 = 26 years in total."""
    # time for folder name
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    collect = {} # collect val metrics for final csv summary
    val_year_start = 1996 + args.init_train_length
    val_year_end = val_year_start + args.val_length - 1
    best_ckpt_path = None # for both train and tune
    best_config = None # for tune
    for year_idx in range(27 - (args.init_train_length + args.val_length + args.test_length)):
    # for year_idx in range(1):
        collect[f"val{val_year_start+year_idx}{val_year_end+year_idx}"], \
        summary_path, \
        best_ckpt_path, \
        best_config, \
        = run(args, year_idx, time, best_ckpt_path, best_config)

    # calculate mean, std and save to .csv
    summary_to_csv(collect, summary_path)

def add_stress_test_param(args):
    """Stress test model with args that give largest dataset."""
    # Get dictionary of args.
    d = vars(args)
    # One loop of maximum data size. Can edit args via dictionary.
    d["init_train_length"] = 26 - args.val_length - args.test_length
    d["dataset"] = "big"
    d["max_epochs"] = 3
    d["num_samples"] = 1


if __name__ == "__main__":

    parser = ArgumentParser(description=
    "Master Thesis Mathias Ruoss - Option Return Classification: "
    "Loop over all train, val splits and predict on test set for given model.")
    subparsers = parser.add_subparsers()

    # Define subparser level arguments.
    parser_train = subparsers.add_parser("train")
    parser_train.set_defaults(mode="train") # string can also be a function
    parser_tune = subparsers.add_parser("tune") 
    parser_tune.set_defaults(mode="tune") # string can also be a function

    parser_train.add_argument("model", choices=["nn"])
    parser_tune.add_argument("model", choices=["nn", "lin", "svm", "xgb"])

    # Parse mode and model first to determine which args to load in load_args.
    args_, _ = parser.parse_known_args()

    # Load args only for the specified mode and model.
    load_args(locals()[f"parser_{args_.mode}"], 
                args_.mode, args_.model)
    
    # Set general model agnostic hyperparams.
    cockpit = parser.add_argument_group("Loop Configuration")
    cockpit.add_argument("--seed", type=int, default=42)
    cockpit.add_argument("--path_data", type=str, default= r"C:\Users\Mathiass\OneDrive - Universität Zürich UZH\Documents\mt_literature\data")
    cockpit.add_argument("--dataset", type=str, default="small")
    cockpit.add_argument("--init_train_length", type=int, default=10)
    cockpit.add_argument("--val_length", type=int, default=2)
    cockpit.add_argument("--test_length", type=int, default=1)
    cockpit.add_argument("--label_fn", type=str, default="binary", 
                            choices=["binary", "multi"])
    cockpit.add_argument("--max_epochs", type=int, default=2) #max_iters for lin, svm, xgb
    cockpit.add_argument("--no_predict", action="store_true") #default: predict
    cockpit.add_argument("--refit", action="store_true") #default: no refit
    cockpit.add_argument("--stress_test", action="store_true")

    args = parser.parse_args()

    # Stress test with biggest dataset? CAREFUL: needs a lot of memory!
    if args.stress_test:
        add_stress_test_param(args)

    seed_everything(args.seed, workers=True)

    looper(args)
