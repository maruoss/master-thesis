import json
from pathlib import Path
import pdb
from shutil import rmtree
from joblib import Memory
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorboard import summary
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from datamodule_loop import Dataset
from sklearn.utils.class_weight import compute_class_weight
from utils.helper import get_best_score, set_tune_log_dir
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from utils.logger import serialize_args, serialize_config

import time as t


def rf_run(args, year_idx, time, ckpt_path, config):
    data = Dataset(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        test_length=args.test_length,
        label_fn=args.label_fn,
    )

    # Get merged datasets (train+val) and train val cv split for sklearn/tune gridsearchcv.
    X_train_val, y_train_val, train_val_split = data.get_cv_data()

    # Load classifier and specified param grid.
    clf, parameter_grid = load_rf(args, data)

    val_year_end, loop_dir, exp_dir = \
        set_tune_log_dir(args, year_idx, time, parameter_grid)

    scheduler = ASHAScheduler(
        max_t=args.n_estimators,
        grace_period= args.grace_pct*args.n_estimators, #how many epochs for sure.
        reduction_factor=args.reduction_factor
    )

    # Randomized Search, not Gridsearch as for sk-classifiers...
    tune_search = TuneSearchCV(
        clf,
        parameter_grid,
        cv=train_val_split,
        early_stopping=scheduler, # early stopping with ASHA
        max_iters=args.n_estimators, # overrules max_iter of SGD classifiers
        scoring=["accuracy", "balanced_accuracy"],
        mode="max",
        refit="balanced_accuracy",
        n_jobs=args.njobs, #how many trials in parallel
        verbose=2,
        local_dir=exp_dir,
        name=loop_dir,
        n_trials=args.num_samples,
        random_state=args.seed,
    #     return_train_score=True # can be comp. expensive
    )

    start = t.time()
    tune_search.fit(X_train_val, y_train_val)
    end = t.time()
    print("Tune GridSearch Fit Time:", end - start)
    
    # Get best scores of gridsearch in a dict.
    best_result = get_best_score(tune_search)
    best_config = tune_search.best_params_

    # Loop Path for best_config and prediction.csv.
    loop_path = Path(Path.cwd(), exp_dir, loop_dir)

    # Add config to val_summary and save best config as .json.
    best_result.update(best_config)
    with open(loop_path/"best_config.json", 'w') as f:
        json.dump(serialize_config(best_config), fp=f, indent=3)

    if not args.no_predict:
        # Automatically predicts with best (refitted) estimator of GridSearchCV
        preds = tune_search.predict(data.X_test)
        preds_df = pd.DataFrame(preds, columns=["pred"])
        test_year_end = val_year_end + args.test_length
        # Prediction directory path.
        save_to_dir = loop_path/f"prediction{test_year_end}.csv"
        preds_df.to_csv(save_to_dir, index_label="id")

    # memory.clear(warn=False)
    # rmtree("cachedir")

    return best_result, exp_dir, ckpt_path, config 


def load_rf(args, data):
    """Load Random Forest Classifier and corresponding parameter grid"""

    # pca
    pca = PCA(random_state=args.seed)

    clf = RandomForestClassifier(
        random_state=args.seed,
    #     class_weight="balanced", #string method not possible in ray tune
        class_weight=data.class_weights,
        n_estimators=args.n_estimators,
    )

    if args.pca:
        pipe = [('pca' , pca), ('clf', clf)]
    else:
        pipe = [('clf', clf)]

    # location = "cachedir"
    # memory = Memory(location=location, verbose=10)

    clf = Pipeline(
        pipe,
        # memory=memory,
        )

    # Example parameters to tune from SGDClassifier
    parameter_grid = {
        "clf__max_depth": tune.randint(1, 6),
        "clf__min_samples_split": tune.choice([2, 3, 4, 5]),
        "clf__min_samples_leaf": tune.choice([1, 2, 3, 4, 5]),
        "clf__max_features": tune.uniform(0.1, 0.3), #sqrt(features) is a good start
        "clf__max_leaf_nodes": tune.choice([2, 3, 4, 5]),
    }
    if args.pca and args.dataset=="small":
        parameter_grid["pca__n_components"] = [None, 5, 10] # None equals no PCA
    elif args.pca and args.dataset=="big":
        parameter_grid["pca__n_components"] = [None, 50, 100] # None equals no PCA

    
    return clf, parameter_grid