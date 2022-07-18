
from pathlib import Path
import pandas as pd


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