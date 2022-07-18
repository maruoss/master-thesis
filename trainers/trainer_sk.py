import json
from pathlib import Path
import pdb
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorboard import summary
from tune_sklearn import TuneGridSearchCV
from datamodule_loop import Dataset
from sklearn.utils.class_weight import compute_class_weight
from utils.helper import get_best_score
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
        # batch_size=args.batch_size,
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        label_fn=args.label_fn,
    )

    # get merged datasets (train+val) and train val cv split for sklearn/tune gridsearchcv
    X_train_val, y_train_val, train_val_split = data.get_cv_data()

    # load scikit classifier and specified param grid
    clf, parameter_grid = load_skmodel_and_param(args, data)

    log_dir = f"./logs/tune/{args.model}_loops"
    train_year_end = 1996 + args.init_train_length + year_idx - 1
    val_year_end = train_year_end + args.val_length
    years = f"train{train_year_end}_val{val_year_end}"
    name = time+"\\"+years

    # save parameter grid as .json
    summary_path = Path.cwd()/log_dir/time
    summary_path.mkdir(exist_ok=True, parents=True)
    with open(summary_path/"config.json", 'w') as f:
        json.dump(serialize_config(parameter_grid), fp=f, indent=3)

    # save args to json
    args_dict = serialize_args(args.__dict__) #functions are not serializable
    with open(summary_path/'args.json', 'w') as f:
        json.dump(args_dict, f, indent=3)


    tune_search = TuneGridSearchCV(
        clf,
        parameter_grid,
        cv=train_val_split,
        early_stopping=True, # early stopping of ASHA
        max_iters=args.max_iters,
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
    
    
    best_result = get_best_score(tune_search)
    config = tune_search.best_params_

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
    
    # scaling is needed in log. reg.
    scaler = StandardScaler()

    # pca
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
        n_components=100, #default 100 
        n_jobs=-1
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

    clf = Pipeline(pipe)

    # Example parameters to tune from SGDClassifier
    parameter_grid = {
        "clf__alpha": [1e-6, 1e-3, 1, 100, 10000],
        "nystroem__n_components": [150, 300], # the higher the better the approx.
    }
    if args.pca and args.dataset=="small":
        parameter_grid["pca__n_components"] = [None, 50, 100] # None equals no PCA
    elif args.pca and args.dataset=="big":
        parameter_grid["pca__n_components"] = [None, 50, 100] # None equals no PCA

    
    return clf, parameter_grid