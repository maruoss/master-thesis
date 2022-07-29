import json
from pathlib import Path
import pdb
from shutil import rmtree
from joblib import Memory
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorboard import summary
from tune_sklearn import TuneGridSearchCV
from datamodule_loop import Dataset
from sklearn.utils.class_weight import compute_class_weight
from utils.helper import get_best_score, set_tune_log_dir
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem

from utils.logger import serialize_args, serialize_config

import time as t


def sk_run(args, year_idx, time, ckpt_path, config):

    # get dataset
    data = Dataset(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        label_fn=args.label_fn,
    )

    # get merged datasets (train+val) and train val cv split for sklearn/tune gridsearchcv
    X_train_val, y_train_val, train_val_split = data.get_cv_data()

    # load scikit classifier and specified param grid
    clf, parameter_grid = load_skmodel_and_param(args, data)

    log_dir, val_year_end, name, summary_path = \
        set_tune_log_dir(args, year_idx, time, parameter_grid)

    tune_search = TuneGridSearchCV(
        clf,
        parameter_grid,
        cv=train_val_split,
        early_stopping=True, # early stopping with ASHA
        max_iters=args.max_epochs, # overrules max_iter of SGD classifiers
        scoring=["accuracy", "balanced_accuracy"],
        refit="balanced_accuracy",
        n_jobs=args.n_jobs, #how many trials in parallel
        verbose=2,
        local_dir=log_dir,
        name=name,
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
    loop_path = Path(Path.cwd(), log_dir, name)

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

    return best_result, summary_path, ckpt_path, config 


def load_skmodel_and_param(args, data):
    if args.model == "lin":
        return load_lin(args, data)
    elif args.model == "svm":
        return load_svm(args, data)
    else:
        raise NotImplementedError("Sk model not implemented.")


def load_lin(args, data):
    """Load logistic regression and corresponding parameter grid"""
    
    # Scaling is needed in log. reg.
    scaler = StandardScaler()

    # Dimension reduction.
    pca = PCA(random_state=args.seed)

    clf = SGDClassifier(
        random_state=args.seed,
    #     class_weight="balanced", #string method not possible in ray tune
        class_weight=data.class_weights,
        max_iter=1, # seems to not matter if tunesearch has max_iters declared
    #     tol=10000000000000000, # seems to be ignored in tunesearchcv
    #     n_iter_no_change=1,  # seems to be ignored in tunesearchcv
    )

#     clf = LogisticRegression(random_state=0, 
#                          class_weight="balanced",
#                            max_iter=1000,
# #                          n_jobs=-1,
# #                          C=0.0001,
#                         )
    if args.pca:
        pipe = [('scaler', scaler), ('pca' , pca), ('clf', clf)]
    else:
        pipe = [('scaler', scaler), ('clf', clf)]

    clf = Pipeline(pipe)

    # Example parameters to tune from SGDClassifier
    parameter_grid = {
        # Dont change loss.
        "clf__loss": [args.loss], #logloss (Log Reg.) or hinge (linear SVM)
        "clf__alpha": [1e-6, 1e-3, 1, 100, 10000],
    }
    if args.pca and args.dataset=="small":
        parameter_grid["pca__n_components"] = [None, 5, 10] # None equals no PCA
    elif args.pca and args.dataset=="big":
        parameter_grid["pca__n_components"] = [None, 50, 100] # None equals no PCA

    
    return clf, parameter_grid


def load_svm(args, data):
    """Load Nystroem Approx and Ker and corresponding parameter grid"""
    
    # scaling is needed in log. reg.
    scaler = StandardScaler()

    # pca
    pca = PCA(random_state=args.seed)

    # Kernel approx. uses RBF Kernel by default
    nystroem = Nystroem(
        # gamma=0.2, 
        random_state=args.seed, 
        # n_components=100, #default 100 
        # n_jobs=-1 # default 1 seems to be faster?
    )

    clf = SGDClassifier(
        random_state=args.seed,
        loss="hinge", #fix to hinge for SVM
    #     class_weight="balanced", #string method not possible in ray tune
        class_weight=data.class_weights,
        max_iter=1, # seems to not matter if tunesearch has max_iters declared
    #     tol=10000000000000000, # seems to be ignored in tunesearchcv
    #     n_iter_no_change=1,  # seems to be ignored in tunesearchcv
    )

    if args.pca:
        pipe = [('scaler', scaler), ('nystroem', nystroem), ('pca' , pca), ('clf', clf)]
    else:
        pipe = [('scaler', scaler), ('nystroem', nystroem), ('clf', clf)]

    # location = "cachedir"
    # memory = Memory(location=location, verbose=10)

    clf = Pipeline(
        pipe,
        # memory=memory,
        )

    # Example parameters to tune from SGDClassifier
    parameter_grid = {
        "clf__alpha": [1e-6, 1e-3, 1, 100, 10000],
        "nystroem__n_components": [50], # the higher the better the approx.
        # 100 or 150 will OOM even on remote server...
    }
    if args.pca and args.dataset=="small":
        parameter_grid["pca__n_components"] = [None, 50, 100] # None equals no PCA
    elif args.pca and args.dataset=="big":
        parameter_grid["pca__n_components"] = [None, 50, 100] # None equals no PCA

    
    return clf, parameter_grid