
import json
from pathlib import Path
import pandas as pd
from utils.logger import serialize_args, serialize_config


def summary_to_csv(collect:dict, summary_path: Path):
    """nested dict of {"year": {"metrics": ...}} to csv"""
    val_summary = pd.DataFrame(collect, index=get_metric_order(collect)) # collect is dict of dict
    val_summary_floats = val_summary.apply(pd.to_numeric, axis=0, errors="coerce")
    val_summary.insert(loc=0, column="std", value=val_summary_floats.std(axis=1))
    val_summary.insert(loc=0, column="mean", value=val_summary_floats.mean(axis=1))
    val_summary.to_csv(Path(summary_path,"val_summary.csv"))


def get_metric_order(nested_dic: dict):
    """get custom order for row indeces of final summary dataframe"""
    order = list(nested_dic[list(nested_dic.keys())[0]].keys())
    metric_order = ["val_bal_acc", "train_bal_acc", "val_acc", "train_acc", "val_loss", "train_loss"]
    for m in metric_order[::-1]:
        if m in order:
            order.remove(m)
            order.insert(0, m) #insert at the front
    return order


def get_best_score(gs):
    """returns best estimator scores of gridsearch.cv_results_ to dictionary"""
    dic = {}
    dic["val_acc"] = gs.cv_results_["mean_test_accuracy"][gs.best_index_]
    dic["val_bal_acc"] = gs.cv_results_["mean_test_balanced_accuracy"][gs.best_index_]
    
    if "mean_train_accuracy" in gs.cv_results_:
        dic["train_acc"] = gs.cv_results_["mean_train_accuracy"][gs.best_index_]
    if "mean_train_balanced_accuracy" in gs.cv_results_:
        dic["train_bal_acc"] = gs.cv_results_["mean_train_balanced_accuracy"][gs.best_index_]
    
    return dic

def set_tune_log_dir(args, year_idx, time, config):
    """set up paths and save config and args there"""
    # Set logging directory for tune.run
    log_dir = f"./logs/tune/{args.model}_loops"
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
        
    return log_dir, val_year_end, name, summary_path