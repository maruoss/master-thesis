from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pandas.api.types import is_numeric_dtype
from sklearn.dummy import DummyClassifier
from joblib import load
import xgboost as xgb

from datamodule import DataModule, Dataset
from model.neuralnetwork import FFN
from model.transformer import TransformerEncoder
from portfolio.helper_ols import regress_on_constant
from portfolio.helper_perf import aggregate_threshold, check_eoy, get_and_check_min_max_pred, get_class_ignore_dates, get_long_short_df, various_tests, weighted_means_by_column
from utils.preprocess import YearMonthEndIndeces, binary_categorize, multi_categorize


def get_yearidx_bestmodelpaths(exp_path: Path, model_name: str) -> list:
    """Search exp_path for 'best_ckpt????' files and save their paths into a list.
    Then sort the path and enumerate it to get a list of tuples (idx, best_ckpt_path)
    for each folder (year) found.

        Returns:
            idx_bestmodelpaths (list): Enumerated best checkpoint paths in the
                                        experiment path.
    """
    #Get pattern for filename (Unix shell-style wildcards)
    pattern = get_filename_bestmodel(model_name) 
    best_ckpt_paths = []
    for directory in exp_path.iterdir():
        if directory.is_dir() and directory.name != "predictions" and directory.name != "portfolios":
            # See https://docs.python.org/3/library/fnmatch.html#module-fnmatch
            # for filename pattern matching below.
            for file in directory.glob(pattern):
                # If files do not exist in 'predictions' folder yet
                best_ckpt_paths.append(file.resolve())
    # IMPORTANT: Sort best ckpt paths that were read in, in ascending order.
    best_ckpt_paths = sorted(best_ckpt_paths, key=lambda x: int(str(x)[-4:]))
    # Append corresponding year_idx to each best_model_path.
    idx_bestmodelpaths = list(enumerate(best_ckpt_paths))
    # Check if year_idx and bestckpts are in the correct order.
    prev_year = -9999
    for yearidx, bestckpt_path in idx_bestmodelpaths:
        if yearidx == 0:
            year = int(bestckpt_path.stem[-4:])
            if not (year > prev_year):
                raise ValueError("Bestmodelpath year is not a positive integer.")
            prev_year = year
        else:
            year = int(bestckpt_path.stem[-4:])
            if not (year == prev_year + 1):
                raise ValueError("bestmodelpaths is either not ordered or years "
                "are missing (not in close succession, eg. 2006->2007->2008, etc.")
            prev_year = year
    return idx_bestmodelpaths


def get_filename_bestmodel(model_name: str) -> str:
    """Since the linear, svm and rf model do not save checkpoints they are not 
    called 'best_ckpt{year}' but best_est{year}. ???? are Unix shell-style 
    wildcards, "which are not the same as regular expressions"
    Source: https://docs.python.org/3/library/fnmatch.html#module-fnmatch. """
    if model_name in ["lin", "svm", "rf"]:
        return "best_est????"
    else:
        return "best_ckpt????"


def check_y_classification(y: np.array, orig_feature_target: pd.DataFrame, label_fn: str) -> None:
    """Checks whether y_data corresponds to y_orig, where y_orig is calculated from
    the classified 'option_ret' column in the original feature target dataframe."""
    y_orig = orig_feature_target["option_ret"] # pd.Series
    # Classify returns (floats) into classes.
    if label_fn == "binary":
        y_orig = y_orig.apply(binary_categorize)
    elif label_fn == "multi3":
        y_orig = y_orig.apply(multi_categorize, classes=3)
    elif label_fn == "multi5":
        y_orig = y_orig.apply(multi_categorize, classes=5)
    else:
        raise ValueError("Specify label_fn as either 'binary' or 'multi'")
    if (len(y) == len(y_orig)) and ((y == y_orig.values).all()):
        print("y_new_year target vector seems to be correct.")
    else:
        raise ValueError("'y_orig' and 'y_new_year' do not have equal values.")


def aggregate_newpred(preds_concat_df: pd.DataFrame, 
                    option_ret_to_agg: pd.DataFrame, # Is used for the data aggregation/ indeces check.
                    args_exp: pd.Series,
                    longclass: int,
                    min_pred: int,
                    ) -> pd.Series:
    """Aggregate orig_feature_df into class portfolios depending on the predictions 
    made in preds_concat_df.
    """
    dates = option_ret_to_agg["date"]
    eoy_indeces, eom_indeces = YearMonthEndIndeces(
                                dates=dates, 
                                init_train_length=args_exp["init_train_length"],
                                val_length=args_exp["val_length"],
                                test_length=args_exp["test_length"]
                                ).get_indeces()
    # Slice df_small to prediction period.
    # Get first month of first year of eventual predictions.
    preds_start_idx = list(eom_indeces.values())[0][0]
    option_ret_to_agg = option_ret_to_agg.iloc[preds_start_idx:]
    # Make sure df_small and preds_concat_df are of same length.
    assert len(preds_concat_df) == len(option_ret_to_agg), ("length of prediction dataframe "
                                    "is not equal the sliced option return dataframe")
    # Align indeces with preds_concat_df, but dont drop old index.
    option_ret_to_agg = option_ret_to_agg.reset_index(drop=False)
    # Concatenate option return data and predictions.
    concat_df = pd.concat([option_ret_to_agg, preds_concat_df], axis=1)
    # Checks whether rows with id of 0 correspond to start of new years.
    assert check_eoy(concat_df, eoy_indeces), ("Id 0 and eoy indeces do not match.")
    # Set df_small index back to main index.
    concat_df = concat_df.set_index("index", drop=True)
    print("Done.")

    # Create single weight column 'if_long_short' with -1 for lowest and 1 for 
    # highest predicted class. Rest is 0.
    print("Create weight columns for each class...")
    max_pred, min_prediction, classes = get_and_check_min_max_pred(concat_df, args_exp["label_fn"])
    # 1.5x faster than pd.map...
    condlist = [concat_df["pred"] == min_prediction, concat_df["pred"] == longclass]
    choicelist = [-1, 1]
    no_alloc_value = 0
    concat_df["if_long_short"] = np.select(condlist, choicelist, no_alloc_value)
    # Create separate weight columns for each class in concat_df.
    for c in classes:
        condlist = [concat_df["pred"] == c]
        choicelist = [1]
        no_alloc_value = 0
        concat_df[f"weights_{c}"] = np.select(condlist, choicelist, no_alloc_value)

    # Only calculate weighted average for numerical columns (have to drop 'date').
    col_list = [val for val in concat_df.columns.tolist() if "date" not in val]
    print("Done.")
    # Aggregate and collect all portfolios in a dictionary with key 'class0', 'class1', etc.
    print("Aggregate for each class and collect the dataframes...")
    # Aggregate returns per month where there are at least 'min_pred' predictions 
    # made for that class in that month, otherwise return 0 for that month.
    agg_dict = aggregate_threshold(
                                concat_df=concat_df, 
                                classes=classes, 
                                col_list=col_list, 
                                agg_func=weighted_means_by_column,
                                min_pred=min_pred)
    print("Done.")
    
    print("Which classes were not invested in at all in the respective month?...")
    # For each class print out months where no prediction was allocated for that class, 
    # and save these indeces for short and long class to later ignore the returns of 
    # these months.
    class_ignore = get_class_ignore_dates(agg_dict, classes, longclass=longclass) #returns dict
    print("Done.")
    
    # Perform various tests to check our calculations.
    test_concat = concat_df.copy()
    test_agg_dict = agg_dict.copy()
    print("***Sanity test the aggregated results...***")
    various_tests(agg_dict, concat_df, col_list, classes, class_ignore, 
                    min_pred=min_pred, longclass=longclass)
    # Make sure tests did not alter dataframes.
    pd.testing.assert_frame_equal(test_concat, concat_df)
    for c in classes:
        pd.testing.assert_frame_equal(test_agg_dict[f"class{c}"], agg_dict[f"class{c}"])
    print("Done.")
    print("********************************************")

    print("Create Long Short Portfolio while ignoring months where one side "
        "is not allocated...")
    shortclass = classes[0] #should be 0
    assert shortclass == 0, "Class of short portfolio not 0. Check why."
    print(f"Subtract Short portfolio (class {shortclass}) from Long portfolio "
            f"(class {longclass})...")
    # Long-Short PF (highest class (long) - lowest class (short))
    long_short_df = get_long_short_df(agg_dict, classes, class_ignore, 
                                    shortclass=shortclass, longclass=longclass)
    print("Done.")
    return long_short_df["option_ret"]


def pred_on_data(
        model_name: str, 
        yearidx: int, 
        bestmodelpath: Path, 
        permuted_feature_target_df: pd.DataFrame,
        args_exp: pd.Series,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    # Models based on Pytorch Lightning.
    if model_name in ["nn", "transformer"]:
        # If small dataset can predict in one go, otherwise make batchsize smaller.
        if args_exp.dataset == "small":
            batch_size = 10000 #transformer needs small(er) batch size here as well.
        else:
            batch_size = 1000
        if model_name == "nn":
            model = FFN.load_from_checkpoint(bestmodelpath)
        elif model_name == "transformer":
            model = TransformerEncoder.load_from_checkpoint(bestmodelpath)
        dm = DataModule(
            path=None, #Set to None.
            year_idx=yearidx, #Important.
            dataset=None, #Set to None.
            batch_size=batch_size, #Predict in one step.
            init_train_length=args_exp.init_train_length,
            val_length=args_exp.val_length,
            test_length=args_exp.test_length,
            label_fn=args_exp.label_fn,
            custom_data=permuted_feature_target_df, #Provide data directly.
        )
        trainer = pl.Trainer(
            deterministic=True,
            gpus=1, #gpu fixed to be one here.
            logger=False, #deactivate logging for prediction
            enable_progress_bar=False,
        )
        # Predict.
        preds = trainer.predict(model=model, datamodule=dm) #Returns list of batch predictions.
        preds = torch.cat(preds) #preds is a list already of [batch_size, num_classes]. 
        preds_argmax = preds.argmax(dim=1).numpy()
        preds_perm_df = pd.DataFrame(preds_argmax, columns=["pred"])
        # Dummy predictions.
        dummy_clf = DummyClassifier(strategy="most_frequent")
        preds_dummy = dummy_clf.fit(dm.X_train, dm.y_train).predict(dm.X_test)
        preds_dummy_df = pd.DataFrame(preds_dummy, columns=["pred"])
        # y_true (of the whole data available, not only test dates).
        y = dm.y.numpy()
        return preds_perm_df, preds_dummy_df, y
    elif model_name in ["lin", "svm", "rf", "xgb"]:
        if model_name == "xgb":
            best_bst = xgb.Booster()
            best_bst.load_model(bestmodelpath)
        else:
            best_est = load(bestmodelpath) #Load joblib best estimator file.
        data = Dataset(
            path=None, #Set to None.
            year_idx=yearidx, #Important.
            dataset=None, #Set to None.
            init_train_length=args_exp.init_train_length,
            val_length=args_exp.val_length,
            test_length=args_exp.test_length,
            label_fn=args_exp.label_fn,
            custom_data=permuted_feature_target_df, #Provide data directly.
        )
        # Predict
        if model_name == "xgb":
            # Get test data and labels.
            X_test, y_test = data.get_test()
            D_test = xgb.DMatrix(X_test, label=y_test)
            # Binary problem outputs probabilites, multiclass outputs argmax already.
            preds = best_bst.predict(D_test) #returns np.array
            if data.num_classes == 2:
                preds = np.round(preds) # if 2 classes, round the probabilities
            preds = preds.astype(int) #convert classes from floats to ints.
            preds_perm_df = pd.DataFrame(preds, columns=["pred"])
        else:
            preds = best_est.predict(data.X_test)
            preds_perm_df = pd.DataFrame(preds, columns=["pred"])
        # Dummy predictions.
        dummy_clf = DummyClassifier(strategy="most_frequent")
        preds_dummy = dummy_clf.fit(data.X_train, data.y_train).predict(data.X_test)
        preds_dummy_df = pd.DataFrame(preds_dummy, columns=["pred"])
        # y_true (of the whole dataset)
        y = data.y
        return preds_perm_df, preds_dummy_df, y
    else:
        raise NotImplementedError(f"Model name: '{model_name}' is not implemented yet.")


def mean_str(col):
    """Takes mean of numeric columns in a dataframe, but handles string 
    columns by returning the count of signif. stars '*'. """
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.str.count("\*").mean()


def mean_str_add_stars(df) -> dict:
    """Take mean of columns of df. '*' will be counted
    in string columns and the mean and the corresponding number of
    '*' will be added in a new column.
    """
    series_mean = df.apply(mean_str)
    num_stars = int(series_mean["Signif."])
    series_mean["SignifStars"] = num_stars * "*"
    return series_mean.to_dict()


def preserve_col_order(columns_orig: list, columns_norm: list, only_signif: bool) -> list:
    """Replace string in columns_orig with strings in normalized column names
    that start with the string in colums_orig."""
    column_order = []
    for col_orig in columns_orig:
        if not col_orig.startswith("Mean") and not col_orig.startswith("Mom"):
            to_add = [i for i in columns_norm if i.startswith(col_orig)]
            if only_signif: #If no details, only take significance stars column.
                to_add = [i for i in to_add if "Signif" in i]
            column_order += to_add
        else:
            column_order += [col_orig]
    return column_order


def prepare_to_save(results: dict, only_signif: bool = False) -> pd.DataFrame:
    """Normalizes deeply nested 'results' dictionary to a clean DataFrame. 
    
    First level keys will be row indeces, while second level keys will be the 
    first entry in the column MultiIndex in the final DataFrame."""
    features = list(results.keys())
    measures = list(results[features[0]].keys())
    # For balanced accuracy, json_normalize moves all ols results dictionarys 
    # automatically to the end of the dataframe columns. Need to reorder 
    # normalized df to original order before concatenating all dataframes.
    balacc_columns_orig = list(results[features[0]][measures[0]].keys())
    balacc_columns_norm = list(pd.json_normalize(results[features[0]][measures[0]]).columns)
    balacc_col_order = preserve_col_order(balacc_columns_orig, 
                                        balacc_columns_norm,
                                        only_signif=only_signif)
    # For the portfolio returns, order is already fine (only one ols result at the end). 
    # But columns are also changed if only significance column is desired.
    pfret_momdiff_orig = list(results[features[0]][measures[1]].keys())
    pfret_momdiff_norm = list(pd.json_normalize(results[features[0]][measures[1]]).columns)
    pfret_momdiff_col_order = preserve_col_order(pfret_momdiff_orig,
                                                pfret_momdiff_norm,
                                                only_signif=only_signif)
    feature_collect = []
    for feature in features:
        measure_collect = []
        for measure in measures:
            normalized_df = pd.json_normalize(results[feature][measure])
            if measure == measures[0]: #if measure is about balanced accuracy, keep ols results at original column order.
                normalized_df = normalized_df.reindex(columns=balacc_col_order)
            elif measure == measures[1]: #if measure is about portfolio return, need to keep only signif column if desired.
                normalized_df = normalized_df.reindex(columns=pfret_momdiff_col_order)
            measure_collect.append(normalized_df)
        measure_collect = pd.concat(measure_collect, keys=measures, axis=1)
        feature_collect.append(measure_collect)
    feature_collect = pd.concat(feature_collect, axis=0)
    feature_collect.index = features #set rows to be the feature names.
    return feature_collect


def get_mean_ols_diff(bal_acc_scores: dict, key: str) -> Tuple[dict, dict]:
    """Regresses the ("Diff...") key in the bal_acc_scores dict on the intercept and
    saves ols result in a nice dictionary. Also outputs mean of the key in
    the bal_acc_scores dict.

    Note: Key (string) has to start with a 'Diff' prefix. 
    """
    diff_bal_acc_score = bal_acc_scores[key]
    diff_bal_acc_ols = regress_on_constant(diff_bal_acc_score)
    diff_bal_acc_ols = {key[4:]: {key: values.to_dict() for (key, values) 
                            in diff_bal_acc_ols.items()}}
    mean_diff = {f"Mean{key}": np.mean(diff_bal_acc_score)}
    return mean_diff, diff_bal_acc_ols


def sanity_check_balacc_means(mean_bal_acc_ols: dict) -> None:
    """Checks whether different numbers for the means coincide in value."""
    for key in mean_bal_acc_ols.keys():
        if not key.startswith("Mean"): # Only look at OLS result dictionaries.
            for se in mean_bal_acc_ols[key].keys():
                if not abs(mean_bal_acc_ols[key][se]["Coef."]["const"] - mean_bal_acc_ols[f"MeanDiff{key}"]) < 0.0000001:
                    raise ValueError(f"Bal. acc. mean and the OLS coefficient are not equal for key {key}.")
    if not abs(mean_bal_acc_ols["MeanOrig"] - mean_bal_acc_ols["MeanPerm"] - mean_bal_acc_ols["MeanDiffOrigPerm"]) < 0.0000001:
        raise ValueError("Mean Orig - Mean Permu is not equal to the mean difference.")
    if not abs(mean_bal_acc_ols["MeanOrig"] - mean_bal_acc_ols["MeanDummy"] - mean_bal_acc_ols["MeanDiffOrigDummy"]) < 0.0000001:
        raise ValueError("Mean Orig - Mean Dummy is not equal to the mean difference.")
    if not abs(mean_bal_acc_ols["MeanPerm"] - mean_bal_acc_ols["MeanDummy"] - mean_bal_acc_ols["MeanDiffPermDummy"]) < 0.0000001:
        raise ValueError("Mean Perm - Mean Dummy is not equal to the mean difference.")

def sanity_check_mom_ls_means(mom_ls_ret_mean_ols: dict) -> None:
    """Checks whether different numbers for the mean of means coincide in value."""
    for i in mom_ls_ret_mean_ols.keys():
        if not i.startswith("Mom"):
            if not abs(mom_ls_ret_mean_ols[i]["Coef."] - mom_ls_ret_mean_ols["MomDiff"]) < 0.0000001:
                raise ValueError(f"Bal. acc. mean and the OLS coefficient are not equal for key {i}.")
            if not abs(mom_ls_ret_mean_ols["MomOrig"] - mom_ls_ret_mean_ols["MomNew"] - mom_ls_ret_mean_ols["MomDiff"]) < 0.0000001:
                raise ValueError("Mean of mean (Mom) Orig - Mean of mean (Mom) New is not equal to the mean of mean difference.")

def sanity_check_ls_means(
        diff_ols_results, 
        ls_pf_ret_diff_mean,
        ls_pf_ret_orig_mean,
        ls_pf_ret_perm_mean
        ) -> None:
    """Checks whether different numbers for the means coincide in value."""
    for i in diff_ols_results.keys():
        if not abs(diff_ols_results[i]["Coef."].item() - ls_pf_ret_diff_mean) < 0.0000001:
            raise ValueError(f"Mean and OLS coefficient are not equal for key {i}.")
        if not abs(ls_pf_ret_orig_mean - ls_pf_ret_perm_mean - ls_pf_ret_diff_mean) < 0.0000001:
            raise ValueError("Mean Orig - Mean New is not equal to the mean difference.")