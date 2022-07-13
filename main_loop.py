from pathlib import Path
import pdb
from datetime import datetime
import pandas as pd

from pytorch_lightning import seed_everything
from argparse import ArgumentParser
from utils.arguments import load_args
from trainers.trainer_nn import nn_train, nn_tune



def train(args, year_idx, time, ckpt_path):
    """single loop with fixed hyperparameters"""
    if args.model == "nn":
        best_result, summary_path, ckpt_path = nn_train(args, year_idx, time, ckpt_path)

    return best_result, summary_path, ckpt_path

def tune(args, year_idx, time, ckpt_path, config):
    """hyperparameter search over a single loop"""
    if args.model == "nn":
        # no checkpoint here -> no refit possible
        best_result, summary_path, ckpt_path, config \
            = nn_tune(args, year_idx, time, ckpt_path, config)

    return best_result, summary_path, ckpt_path, config

def looper(args):
    # Loop over all train, val splits: 1996 - 2021 = 26 years in total
    # time for folder name
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    collect = {} # collect val metrics for final csv summary
    val_year_start = 1996 + args.init_train_length
    val_year_end = val_year_start + args.val_length - 1
    best_ckpt_path = None # for both train and tune
    best_config = None # for tune
    #TODO for year_idx in range(26 - (args.init_train_length + args.val_length)):
    for year_idx in range(2):
        if args.mode == "train":
            collect[f"val{val_year_start+year_idx}{val_year_end+year_idx}"], \
            summary_path, \
            best_ckpt_path, \
            = train(args, year_idx, time, best_ckpt_path)

        elif args.mode == "tune":
            collect[f"val{val_year_start+year_idx}{val_year_end+year_idx}"], \
            summary_path, \
            best_ckpt_path, \
            best_config, \
            = tune(args, year_idx, time, best_ckpt_path, best_config)
        else:
            raise NameError("Choose either train or tune as a mode.")
    
    # calculate mean, std and save to .csv
    val_summary = pd.DataFrame(collect)
    val_summary_floats = val_summary.apply(pd.to_numeric, axis=0, errors="coerce")
    val_summary.insert(loc=0, column="std", value=val_summary_floats.std(axis=1))
    val_summary.insert(loc=0, column="mean", value=val_summary_floats.mean(axis=1))
    val_summary.to_csv(Path(summary_path,"val_summary.csv"))

if __name__ == "__main__":

    parser = ArgumentParser(description="Master Thesis Mathias Ruoss - Option\
        Return Classification")
    subparsers = parser.add_subparsers()

    # subparser level
    parser_train = subparsers.add_parser("train")
    parser_train.set_defaults(mode="train")
    parser_tune = subparsers.add_parser("tune")
    parser_tune.set_defaults(mode="tune")

    # 2. subparser level
    # subsubparser = parser_train.add_subparsers()
    # parser_train_nn = subsubparser.add_parser("nn")
    # parser_train_nn.set_defaults(model="nn")

    # subsubparser = parser_tune.add_subparsers()
    # parser_tune_nn = subsubparser.add_parser("nn")
    # parser_tune_nn.set_defaults(model="nn")

    parser_train.add_argument("model", choices=["nn"])
    parser_tune.add_argument("model", choices=["nn"])

    # parse mode and model first to determine which args  to load in load_args
    args_, _ = parser.parse_known_args()

    # load args only for the specified mode and model
    load_args(locals()[f"parser_{args_.mode}"], 
                args_.mode, args_.model)
    
    # model agnostic hyperparams
    cockpit = parser.add_argument_group("Loop Configuration")
    cockpit.add_argument("--seed", type=int, default=42)
    cockpit.add_argument("--path_data", type=str, default= r"C:\Users\Mathiass\OneDrive - Universität Zürich UZH\Documents\mt_literature\data")
    # tune params (ignore in train)
    cockpit.add_argument("--init_train_length", type=int, default=10)
    cockpit.add_argument("--val_length", type=int, default=2)
    cockpit.add_argument("--test_length", type=int, default=1)
    cockpit.add_argument("--no_predict", action="store_true")
    cockpit.add_argument("--refit", action="store_true")

    args = parser.parse_args()

    # in tune: will determine hyperparams chosen
    seed_everything(args.seed, workers=True)

    looper(args)
