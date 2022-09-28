from pathlib import Path
from datetime import datetime
import pandas as pd

from pytorch_lightning import seed_everything
from argparse import ArgumentParser
from trainers.trainer_rf import rf_run
from trainers.trainer_transformer import transformer_tune
from trainers.trainer_xgb import xgb_tune
from utils.helper import save_time, summary_to_csv
from trainers.trainer_sk import sk_run
from arguments import load_args
from trainers.trainer_nn import nn_tune


def run(args, year_idx, time, ckpt_path, config):
    """Run a model for one train/val split and return output for the next loop.
    
        Arguments:
            args (Namespace):   Input parameters
            year_idx (int):     Loop index
            time (datetime):    Start_time (+ args.tag) for folder name
            ckpt_path (Path):   Path of best checkpoint
            config (dict):      Config of best checkpoint

        Returns:
            Tuple(
            best_result (dict): Metrics dictionary of best results
            exp_dir (Path):     Experiment directory
            ckpt_path (Path):   Where best model is saved
            config (dict):      Config of best model
            )
    """
    fun_dict = {
                # "nn_train": nn_train, 
                "nn_tune": nn_tune, 
                "lin_tune": sk_run,
                "svm_tune": sk_run,
                "rf_tune": rf_run,
                "xgb_tune": xgb_tune,
                "transformer_tune": transformer_tune,
                }

    # If no checkpoint here -> no refit possible.
    best_result, exp_dir, ckpt_path, config \
        = fun_dict[f"{args.model}_{args.mode}"](args, year_idx, time, ckpt_path, config)

    return best_result, exp_dir, ckpt_path, config

def looper(args):
    """Main runner. Loops over specified train/ val splits and saves results."""
    # Time for logging folder name.
    start_time = datetime.now()
    time = start_time.strftime("%Y%m%d%H%M%S") + args.tag # unique experiment folder name.
    collect = {} # collect val metrics for final csv summary in dict
    time_collect = {} # collect duration for each loop and save in dict
    val_year_start = 1996 + args.init_train_length
    val_year_end = val_year_start + args.val_length - 1
    best_ckpt_path = None # for both train and tune
    best_config = None # for tune
    start_loop_time = start_time
    for year_idx in range(5, 27 - (args.init_train_length + args.val_length + args.test_length) - 1):
        (collect[f"val{val_year_start+year_idx}{val_year_end+year_idx}"],
        exp_dir,
        best_ckpt_path,
        best_config) = run(args, year_idx, time, best_ckpt_path, best_config)
        time_collect[f"loop_{year_idx}"], start_loop_time = save_time(start_loop_time)
        # Some memory leak can be avoided by gc collect.
        import gc
        gc.collect()

    # Calculate metrics and save to .csv.
    summary_to_csv(collect, exp_dir)
    end_time = datetime.now()
    # Save time dictionary to csv.
    time_collect["Total Time"] = save_time(start_time)[0] #returns tuple
    time_collect_df = pd.DataFrame(time_collect).T
    time_collect_df.to_csv(Path(exp_dir,"time.csv"))
    print(f"Finished. Completed in {(end_time - start_time).total_seconds()} seconds.")

def add_stress_test_param(args):
    """Overwrite args (Namespace) with the below parameter update."""
    # Get args as a dictionary.
    d = vars(args)
    # We want maximum data size. Can edit args via dictionary.
    d["init_train_length"] = 26 - args.val_length - args.test_length
    # d["dataset"] = "big" #specify yourself.
    # Depending on which epoch key is avail., change it.
    it_keys = ["max_epochs", "max_iters", "n_estimators", "check_val_every", "num_boost_round"]
    for i in it_keys:
        if i in d.keys():
            d[i] = 1 # change to 1 epoch
    d["num_samples"] = 1 #to test parallelism.
    d["gpus_per_trial"] = 1 #to use full gpu.


if __name__ == "__main__":
    parser = ArgumentParser(description=
        "Master Thesis Mathias Ruoss - Option Return Classification: "
        "Loop over all train, val splits and predict on test set for given model."
    )
    subparsers = parser.add_subparsers()

    # Define subparser level arguments.
    # parser_train = subparsers.add_parser("train")
    # parser_train.set_defaults(mode="train") # string can also be a function
    parser_tune = subparsers.add_parser("tune") 
    parser_tune.set_defaults(mode="tune") # string can also be a function

    # parser_train.add_argument("model", choices=["nn"])
    parser_tune.add_argument("model", choices=["nn", "lin", "svm", "rf", "xgb", "transformer"])

    # Parse mode and model first to determine which args to load in load_args.
    args_, _ = parser.parse_known_args()

    # Load args only for the specified mode and model.
    load_args(locals()[f"parser_{args_.mode}"], 
                args_.mode, args_.model)
    
    # Set current working directory. Used in default --path_data.
    cwd = Path.cwd()

    # Set general model agnostic hyperparams.
    cockpit = parser.add_argument_group("Loop Configuration")
    cockpit.add_argument("--tag", type=str, default="")
    cockpit.add_argument("--seed", type=int, default=42)
    cockpit.add_argument("--path_data", type=str, default=cwd/"data")
    cockpit.add_argument("--dataset", type=str, default="medium",
                            choices=["small", "medium", "big"])
    cockpit.add_argument("--init_train_length", type=int, default=10)
    cockpit.add_argument("--val_length", type=int, default=2)
    cockpit.add_argument("--test_length", type=int, default=1)
    cockpit.add_argument("--label_fn", type=str, default="multi5", 
                            choices=["binary", "multi3", "multi5"])
    # cockpit.add_argument("--max_epochs", type=int, default=100) #max_iters for lin, svm, xgb
    cockpit.add_argument("--no_predict", action="store_true") #default: predict
    cockpit.add_argument("--refit", action="store_true") #default: no refit
    cockpit.add_argument("--stresstest", action="store_true")
    # Tune configuration
    cockpit = parser.add_argument_group("Tune Configuration")
    cockpit.add_argument("--num_samples", type=int, default=30, help="How many "
                        "parameter configurations are sampled.")
    cockpit.add_argument("--gpus_per_trial", type=float, default=0.25) #for 1 gpu, gives 4 parallel trials.
    cockpit.add_argument("--njobs", type=int, default=8) #for 16 cpus, 8 parralel jobs use 2 cpus/trial.
    # ASHA
    cockpit.add_argument("--grace_pct", type=float, default=0.2, 
                        help="Percentage of epochs that have to be run before "
                        "early stop is possible.")
    cockpit.add_argument("--reduction_factor", type=int, default=2)

    args = parser.parse_args()

    assert args.init_train_length + args.val_length + args.test_length <= 26, \
            ("(train + val + test) is bigger than total years in dataset (26).")

    # Stress test with biggest dataset? CAREFUL: needs a lot of memory!
    if args.stresstest:
        add_stress_test_param(args)

    seed_everything(args.seed, workers=True)

    looper(args)
