import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from typing import Tuple
import shutil

from datamodule import Dataset
from utils.logger import serialize_config
from utils.helper import del_ckpts, set_tune_log_dir
import ray



def bal_acc_xgb(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label() # returns np.array
    # if binary problem, preds will be probabilites... . Doesnt affect multi problem.
    preds = np.round(preds)
    val_bal_acc = balanced_accuracy_score(y, preds)
    return 'val_bal_acc', val_bal_acc


def inner_xgb_tune(config, args, year_idx, ckpt_path):
    # pl.seed_everything(42, workers=True) #not needed, as long as 'seed' speficied

    # Important: Load dataset inside inner loop. Otherwise, ray will not have
    # enough memory!.
    data = Dataset(
        path=args.path_data,
        year_idx=year_idx,
        dataset=args.dataset,
        # batch_size=args.batch_size,
        init_train_length=args.init_train_length,
        val_length=args.val_length,
        test_length=args.test_length,
        label_fn=args.label_fn,
    )
    # Get datasets from data.
    X_train, X_val, y_train, y_val = data.get_train_val()

    # Adjust metrics depending on binary of multiclass classification.
    metrics = ["logloss", "error"]
    if data.num_classes > 2:
        metrics = ["mlogloss", "merror"]

    es = xgb.callback.EarlyStopping(
        rounds=args.patience, # optimally same as frequency of ChkptCallback
        min_delta=0.0, # 1e-3, 1e-4 too high, set to 0.0 as in NN
        save_best=True, #whether to return best or last model
        maximize=False,
        data_name="eval", #prefix here
        metric_name=metrics[0], #no prefix 'eval' needed
    )

    tc = TuneReportCheckpointCallback(
        metrics={
            "train_logloss": f"train-{metrics[0]}",
            "train_err" : f"train-{metrics[1]}",
            "train_bal_acc": "train-val_bal_acc",
            "val_logloss": f"eval-{metrics[0]}",
            "val_err": f"eval-{metrics[1]}",
            "val_bal_acc": "eval-val_bal_acc",
            # "mean_pred": "mean_pred",
        },
        filename="checkpoint",
        frequency=1, #TODO DOES INCREASING THIS GIVE COMP. SPEED? 
    )

    # Calculate weightings for unbalanced dataset.
    w_array = np.ones(y_train.shape[0])
    for i, val in enumerate(y_train):
        w_array[i] = data.class_weights[val]
    assert len(w_array) == len(y_train) == len(X_train), \
        "Dmatrix weights not the same length as ytrain or X.train"

    # Val weights -> will have an impact on calculating loss -> early stopping.
    w_array_val = np.ones(y_val.shape[0])
    for i, val in enumerate(y_val):
        w_array_val[i] = data.class_weights[val]

    # Build input matrices for XGBoost
    D_train = xgb.DMatrix(X_train, label=y_train, weight=w_array)
    D_val = xgb.DMatrix(X_val, label=y_val, weight=w_array_val)

    # # Delete X_train, X_val, y_train, y_val from memory
    # del X_train, X_val, y_train, y_val # is not reducing memory somehow?

    # Train the classifier.
    results = {}
    bst = xgb.train(
        config,
        D_train,
        evals=[(D_train, "train"), (D_val, "eval")],
        evals_result=results,
        maximize=True,
#         verbose_eval=False,
#         num_boost_round=1, #*************************************************************************
        callbacks=[es, tc],
        custom_metric=bal_acc_xgb, #custom balanced acc. metric
        xgb_model=ckpt_path, #if args.refit=True, this is best model of previous loop
        num_boost_round=args.num_boost_round, # num boost round should equal ASHA max_t
    )
    
    return results

def xgb_tune(args, year_idx, time, ckpt_path, config: dict):

    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "seed": 42,
        "tree_method": "gpu_hist",
        'disable_default_eval_metric': 1,
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.quniform(0.2, 1.0, 0.05),
        "eta": tune.qloguniform(1e-6, 1e-1, 5e-7),
    }

    # In xgb, we have to specify num_classes in advance.
    if args.label_fn == "multi3":
        search_space.update({
            "objective": "multi:softmax",
            "num_class": 3,
            "eval_metric": ["mlogloss", "merror"],
        })
    elif args.label_fn == "multi5":
        search_space.update({
            "objective": "multi:softmax",
            "num_class": 5,
            "eval_metric": ["mlogloss", "merror"],
        })

    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=args.num_boost_round,
        grace_period=args.grace_pct*args.num_boost_round, #how many epochs for sure.
        reduction_factor=args.reduction_factor)
    
    reporter = CLIReporter(max_report_frequency=120, 
                            print_intermediate_tables=True)

    train_fn_with_parameters = tune.with_parameters(inner_xgb_tune,
                                                args=args,
                                                # data=data, # Dont, ray will OOM!
                                                year_idx=year_idx,
                                                ckpt_path=ckpt_path,
                                                )
    resources_per_trial = {
                        # Assuming machine has 8 cores per 1 gpu.
                        "cpu": args.gpus_per_trial * 8,
                        "gpu": args.gpus_per_trial
                        }
    val_year_end, loop_path, exp_path = \
        set_tune_log_dir(args, year_idx, time, search_space)
    
    # Set _temp_dir to folder in same path as experiment, to have control over 
    # disk space, as logging directory may grow considerably...
    # See: https://docs.ray.io/en/latest/ray-core/configure.html#logging-and-debugging.
    ray.init(_temp_dir=str(exp_path/"temp_ray"), ignore_reinit_error=True)
    analysis = tune.run(
        train_fn_with_parameters,
        local_dir=exp_path,
        metric="val_logloss", #TuneReportChkptCallback redefined names
        mode="min",
        # You can add "gpu": 0.1 to allocate GPUs.
        resources_per_trial=resources_per_trial,
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=loop_path,
        fail_fast=True, # stop all trials as soon as any trial errors
        keep_checkpoints_num=1, # only keep best checkpoint
        checkpoint_score_attr="min-val_logloss", #TuneReportChkptCallback redefined names
    )
    print("Best hyperparameters found were: ", analysis.best_config)

    #change "last" to "all" for global min
    best_last_trial = analysis.get_best_trial("val_logloss", "min", "last")
    print("Best trial among last epoch config: {}".format(best_last_trial.config))
    print("Best trial >>last epoch<< validation loss: {}".format(
        best_last_trial.last_result["val_logloss"]))
    print("Best trial >>last epoch<< validation balanced accuracy: {}".format(
        best_last_trial.last_result["val_bal_acc"]))
    
    # Should yield the data for the best checkpoint, if checkpoints are made every epoch.
    best_result_per_trial_df = analysis.dataframe(metric="val_logloss", 
                                                    mode="min").sort_values("val_logloss")
    # Save df to folder?
    best_result = best_result_per_trial_df.iloc[0, :].to_dict() #take best values of best trial

    # Test prediction: best checkpoint out of all trials.
    best_trial = analysis.get_best_trial("val_logloss", "min", scope="all")
    best_config = best_trial.config

    # Loop Path for best_config and prediction.csv.
    loop_path = Path(Path.cwd(), exp_path, loop_path)
    
    # Save best config as .json.
    with open(loop_path/"best_config.json", 'w') as f:
        json.dump(serialize_config(best_config), fp=f, indent=3)

    if args.refit:
        config = best_trial.config
        ckpt_path = Path(analysis.get_best_checkpoint(best_trial).get_internal_representation()[1], 
                        "checkpoint")
    
    if not args.no_predict:
        # Load dataset here, need X_test data.
        data = Dataset(
            path=args.path_data,
            year_idx=year_idx,
            dataset=args.dataset,
            # batch_size=args.batch_size,
            init_train_length=args.init_train_length,
            val_length=args.val_length,
            test_length=args.test_length,
            label_fn=args.label_fn,
        )
        # Load best model.
        best_path = Path(analysis.get_best_checkpoint(best_trial).get_internal_representation()[1],
                        "checkpoint")
        # Copy best model to loop folder for later analysis.
        test_year_end = val_year_end + args.test_length
        new_best_path = loop_path/f"best_ckpt{test_year_end}"
        shutil.copy2(best_path, new_best_path)
        # To save disk space: Delete all other checkpoints (take huge disk space).
        print("Delete checkpoints in trials to save disk space...")
        del_ckpts(loop_path)
        print("Done!")
        print(f"Loading model to predict from path: {new_best_path}")
        best_bst = xgb.Booster()
        best_bst.load_model(new_best_path)

        # Get test data and labels.
        X_test, y_test = data.get_test()
        D_test = xgb.DMatrix(X_test, label=y_test)
        # Predict.
        # Binary problem outputs probabilites, multiclass outputs argmax already.
        preds = best_bst.predict(D_test) #returns np.array
        if data.num_classes == 2:
            preds = np.round(preds) # if 2 classes, round the probabilities
        preds = preds.astype(int) #convert classes from floats to ints.
        # preds_argmax = preds[0].argmax(dim=1).numpy() # assumes batchsize is whole testset
        preds_df = pd.DataFrame(preds, columns=["pred"])
        # Prediction path.
        save_to_dir = loop_path/f"prediction{test_year_end}.csv"
        preds_df.to_csv(save_to_dir, index_label="id")

    return best_result, exp_path, ckpt_path, config