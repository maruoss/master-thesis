from pathlib import Path
import pdb
from datetime import datetime
import pandas as pd

from pytorch_lightning import seed_everything
from argparse import ArgumentParser
from utils.arguments import load_args
from trainers.train_NN import nn_train, nn_tune



def train(args, year_idx, time):
    """single loop with fixed hyperparameters"""
    if args.model == "nn":
        nn_loop(args, year_idx, time)
    else:
        raise ValueError("specify an implemented model")

def tune(args, year_idx, time):
    """hyperparameter search over a single loop"""
    if args.model == "nn":
        nn_tune(args, year_idx, time)
    else:
        raise ValueError("specify an implemented model")

def looper(args):
    # Loop over all train, val splits: 1996 - 2021 = 26 years in total
    # time for folder name
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    for year_idx in range(26 - (args.init_train_length + args.val_length)):
        args.mode(args, year_idx, time)


if __name__ == "__main__":

    parser = ArgumentParser(description="Master Thesis Mathias Ruoss - Option\
        Return Classification")
    subparsers = parser.add_subparsers()

    # subparser level
    parser_train = subparsers.add_parser("train")
    parser_train.set_defaults(mode=train)
    parser_tune = subparsers.add_parser("tune")
    parser_tune.set_defaults(mode=tune)

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
    load_args(locals()[f"parser_{args_.mode.__name__}"], 
                args_.mode.__name__, args_.model)

    # model agnostic hyperparams
    cockpit = parser.add_argument_group("Loop Configuration")
    cockpit.add_argument("--seed", type=int, default=42)
    cockpit.add_argument("--path_data", type=str, default= r"C:\Users\Mathiass\OneDrive - Universität Zürich UZH\Documents\mt_literature\data")
    cockpit.add_argument("--init_train_length", type=int, default=10)
    cockpit.add_argument("--val_length", type=int, default=2)
    cockpit.add_argument("--test_length", type=int, default=1)

    args = parser.parse_args()

    # in tune: will determine hyperparams chosen
    seed_everything(args.seed, workers=True)

    looper(args)
