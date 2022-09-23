from argparse import ArgumentParser
from pathlib import Path
import pdb
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from data.utils.convert_check import small_med_big_eq
from datamodule import load_data
from portfolio.helper_ols import regress_factors, regress_on_constant
from portfolio.helper_perf import check_eoy, collect_preds, concat_and_save_preds, filter_idx, get_and_check_min_max_pred, get_class_ignore_dates, save_performance_statistics, various_tests, weighted_means_by_column
from portfolio.helper_featureimp import (aggregate_newpred, 
                                        check_y_classification, 
                                        get_mean_ols_diff, 
                                        get_yearidx_bestmodelpaths, 
                                        mean_str_add_stars, 
                                        pred_on_data, 
                                        prepare_to_save, 
                                        sanity_check_balacc_means, 
                                        sanity_check_ls_means, 
                                        sanity_check_mom_ls_means
                                        )


from portfolio.load_files import load_ff_monthly, load_mom_monthly, load_pfret_from_pfs, load_rf_monthly, load_vix_monthly, load_vvix_monthly
from utils.preprocess import YearMonthEndIndeces


def aggregate(args):
    """Create aggregated monthly csv files in a new 'portfolios' subfolder within
    the logs/experiment_folder. The aggregation is done by equally weighting
    each option for the respective month."""
    print("Start aggregation of predictions in each month...")
    # Get experiment folder path 'exp_dir'.
    logs_folder = Path.cwd()/"logs"
    matches = Path(logs_folder).rglob(args.expid) #Get folder in logs_folder that matches expid
    matches_list = list(matches)
    if not len(matches_list) == 1:
        raise ValueError(f"There exists none or more than 1 folder with "
                            f"experiment id {args.expid} in the {logs_folder.name} "
                            "directory!")
    exp_dir = matches_list[0]
    # Move all predictions to 'predictions' folder with the expdir folder.
    print("Find all 'predictions{year}' files in the subfolders of the experiment "
    "and copy to 'predictions' folder...")
    collect_preds(exp_dir)
    print("Done.")
    #TODO: combine (ensemble) of multiple experiment predictions together? list of expids?
    # Read all prediction .csv and save as "all_pred.csv" in exp_dir.
    print("Read in all prediction .csv files as a dataframe and save as 'all_pred.csv'...")
    preds_concat_df = concat_and_save_preds(exp_dir) #shape: [index, columns: [id, pred]]
    print("Done.")

    # Get path where datasets reside:
    print("Concat the dataframe with the respective option data...")
    path_data = Path.cwd()/"data"
    # Check whether small, med, big are all equal (whether we predicted on same 
    # test data ordering!)
    # Takes 11 secs., uncomment for final production code.
    # assert small_med_big_eq(datapath), ("Date and return columns NOT equal between "
    #                                 "small, medium and big datasets!")
    # print("Dates and option return columns from small, medium and big datasets are "
    #         "equal!")
    # Get small dataset (irrespective of small/medium/big used for train!).
    df_small = pd.read_parquet(path_data/"final_df_call_cao_small.parquet")
    dates = df_small["date"]
    # Get args from experiment.
    # Alternatively: Load json with json.load() and convert dict/list to df.
    args_exp = pd.read_json(exp_dir/"args.json", typ="series")
    # Get start of year index and all end of month indeces. Load with args that 
    # were used in the actual experiment.
    eoy_indeces, eom_indeces = YearMonthEndIndeces(
                                dates=dates, 
                                init_train_length=args_exp["init_train_length"],
                                val_length=args_exp["val_length"],
                                test_length=args_exp["test_length"]
                                ).get_indeces()
    # Slice df_small to prediction period.
    # Get first month of first year of eventual predictions.
    preds_start_idx = list(eom_indeces.values())[0][0]
    df_small = df_small.iloc[preds_start_idx:]
    # Make sure df_small and preds_concat_df are of same length.
    assert len(preds_concat_df) == len(df_small), ("length of prediction dataframe "
                                    "is not equal the sliced option return dataframe")
    # Align indeces with preds_concat_df, but dont drop old index.
    df_small = df_small.reset_index(drop=False)
    # Concatenate option return data and predictions.
    concat_df = pd.concat([df_small, preds_concat_df], axis=1)
    # Checks whether rows with id of 0 correspond to start of new years.
    assert check_eoy(concat_df, eoy_indeces), ("Id 0 and eoy indeces do not match.")
    # Set df_small index back to main index.
    concat_df = concat_df.set_index("index", drop=True)
    print("Done.")

    # Create single weight column 'if_long_short' with -1 for lowest and 1 for 
    # highest predicted class. Rest is 0.
    print("Create weight columns for each class...")
    max_pred, min_pred, classes = get_and_check_min_max_pred(concat_df, args_exp["label_fn"])
    # 1.5x faster than pd.map...
    condlist = [concat_df["pred"] == min_pred, concat_df["pred"] == max_pred]
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
    agg_dict = {}
    for c in classes:
        agg_df = concat_df.groupby("date").aggregate(weighted_means_by_column, col_list, f"weights_{c}")
        agg_dict[f"class{c}"] = agg_df
    print("Done.")
    
    print("Which classes were not predicted at all in a respective month?...")
    # For each class print out months where no prediction was allocated for that class, 
    # and save these indeces for short and long class to later ignore the returns of 
    # these months.
    class_ignore = get_class_ignore_dates(concat_df, classes) #returns dict
    print("Done.")
    
    # Perform various tests to check our calculations.
    test_concat = concat_df.copy()
    test_agg_dict = agg_dict.copy()
    print("Sanity test the aggregated results...")
    various_tests(agg_dict, concat_df, col_list, classes, class_ignore)
    print("Done.")
    # Make sure tests did not alter dataframes.
    pd.testing.assert_frame_equal(test_concat, concat_df)
    for c in classes:
        pd.testing.assert_frame_equal(test_agg_dict[f"class{c}"], agg_dict[f"class{c}"])

    print("Save each dataframe in the 'portfolios' subfolder...")
    # Save all aggregated dataframes per class to 'portfolios' subfolder within the 
    # experiment directory 'exp_dir'.
    pf_dir = exp_dir/"portfolios"
    try: # raise error if 'portfolio' folder exists already
        pf_dir.mkdir(exist_ok=False, parents=False) # raise error if parents are missing.
        for class_c, df in agg_dict.items():
            df.to_csv(pf_dir/f"{class_c}.csv")
    except FileExistsError as err: # from 'exist_ok' -> portfolios folder already exists, do nothing.
        raise FileExistsError("Directory 'portfolios' already exists. Will not "
        "touch folder and exit code.") from err
    print("Done.")

    print("Create Long Short Portfolio while ignoring months where one side "
        "is not allocated...")
    # Long-Short PF (highest class (long) - lowest class (short))
    short_class = classes[0] #should be 0
    assert short_class == 0, "Class of short portfolio not 0. Check why."
    long_class = classes[-1] #should be 2 for binary, 3 for 'multi3', etc.
    print(f"Subtract Short portfolio (class {short_class}) from Long portfolio "
            f"(class {long_class}) and save to long{long_class}short{short_class}.csv...")
    # Subtract short from long portfolio.
    long_df = agg_dict[f"class{long_class}"].copy() #deep copy to not change original agg_dict
    short_df = agg_dict[f"class{short_class}"].copy() #deep copy to not change original agg_dict
    months_no_inv = class_ignore[f"class{long_class}"].union(class_ignore[f"class{short_class}"]) #union of months to set to 0.
    long_df.loc[months_no_inv, :] = 0
    short_df.loc[months_no_inv, :] = 0
    long_short_df = long_df - short_df #months that are 0 in both dfs stay 0 everywhere.
    assert ((long_short_df.drop(months_no_inv)["pred"] == (long_class - short_class)).all() and #'pred' should be long_class - short_class
            (long_short_df.drop(months_no_inv)["if_long_short"] == 2).all()) #'if_long_short' should be 2 (1 - (-1) = 2)
    # Drop one-hot "weight" columns here.
    cols_to_keep = [col for col in long_short_df.columns.tolist() if "weight" not in col]
    long_short_df = long_short_df[cols_to_keep]
    long_short_df.to_csv(pf_dir/f"long{long_class}short{short_class}.csv")
    print("Done.")
    print("All done!")


def performance(args):
    """Read the monthly aggregated portfolios from the 'portfolios' subfolder
    within the experiment directory and procude performance statistics in a csv,
    png and latex file. Also produce a plot with the performance of each portfolio."""
    print("Starting performance evaluation of portfolios...")
    sns.set_style("whitegrid") # style must be one of white, dark, whitegrid, darkgrid, ticks
    sns.set_context('paper', font_scale=2.0) # set fontsize scaling for labels, axis, legend, automatically moves legend

    # Get experiment folder path 'exp_dir'.
    logs_folder = Path.cwd()/"logs"
    matches = Path(logs_folder).rglob(args.expid) #Get folder in logs_folder that matches expid
    matches_list = list(matches)
    if not len(matches_list) == 1:
        raise ValueError(f"There exists none or more than 1 folder with "
                            f"experiment id {args.expid} in the {logs_folder.name} "
                            "directory!")
    exp_path = matches_list[0]

    # Read aggregated portfolios. RETURNS are from 02/1996 to 11/2021. Because
    # predictions are made at e.g. 31-10-2021, but the return is from the month 11/2021. 
    print("Reading aggregated portfolios from 'portfolios' folder...")
    path_portfolios = exp_path/"portfolios"
    pf_returns = load_pfret_from_pfs(path_portfolios)
    print("Done.")

    # Data path.
    path_data = Path.cwd()/"data"

    print("Load the Monthly Riskfree rate from the 5 Fama French Factors Dataset...")
    # Skip first two rows (text description) and omit yearly data (after row 706).
    monthly_rf = load_rf_monthly(path_data) #dataframe
    # Filter months (rows) to align with pf_returns.
    monthly_rf = filter_idx(monthly_rf, pf_returns)

    # Subtract Rf from class portfolio returns to get excess returns.
    print("Subtract Riskfree rate only from *class* portfolios AND where there was "
          "made an investment, but *not* from the long short portfolio, as the riskfree "
          "rate cancels out for the long short portfolio...")
    class_pf_names = [col for col in pf_returns.columns if col.startswith("class")]
    # ONLY SUBTRACT RF FROM MONTHS IN CLASS PORTFOLIOS WHERE THERE WAS AN INVESTMENT MADE.
    for pf in class_pf_names:
        # Exactly zero return means there was no investment at all (no prediction for that class in that month).
        zero_inv_mask = (pf_returns[pf] == 0)
        # Where replaces values where condition is False (where there was investment -> subtract riskfree of the respective month).
        pf_returns.loc[:, pf] = (pf_returns.loc[:, pf]).where(zero_inv_mask, lambda x: x - monthly_rf.values.squeeze()) #.values needed to broadcast over columns.
    # Make 'results' subfolder.
    path_results = exp_path/"results"
    path_results.mkdir(exist_ok=True, parents=False)
    # Make 'performance' subfolder in 'results' folder.
    path_results_perf = exp_path/"results"/"performance"
    path_results_perf.mkdir(exist_ok=True, parents=False)

    # Save performance statistics in the 'performance' subfolder in exp_dir/'results'.
    # Get experiment args to get modelname for naming files...
    args_exp = pd.read_json(exp_path/"args.json", typ="series")
    print("Saving performance statistics...")
    save_performance_statistics(pf_returns, path_data, path_results_perf, args_exp.model)
    print("Done.")

    # ****** Regressions ******
    print("Explain monthly returns by regressing them on various factors...")
    print("Load the 5 Fama French factors...")
    # Skip first two rows (text description) and omit yearly data (after row 706).
    ff_monthly = load_ff_monthly(path_data)
    print("Done.")

    print("Load the Momentum factor...")
    # Skip first 13 rows (text description) and omit yearly data (after row 1144).
    mom_monthly = load_mom_monthly(path_data)
    print("Done.")

    print("Load the VIX data...")
    vix_monthly = load_vix_monthly(path_data)
    print("Done.")

    print("Load the VVIX data...")
    vvix_monthly = load_vvix_monthly(path_data)
    print("Done.")

    # Variable of interest (y). We want to explain the monthly long short PF returns.
    long_short_pf_returns = pf_returns["long4short0"]

    # Align the months of the following dataframes with the long short dataframe.
    list_to_filter = [vix_monthly, vvix_monthly, ff_monthly, mom_monthly]
    list_to_filter = [filter_idx(i, long_short_pf_returns) for i in list_to_filter]

    # Unpack list to filter
    vix_monthly, vvix_monthly, ff_monthly, mom_monthly = list_to_filter
    # Concat all independent variables to regress them on the long short portfolio.
    factors_avail = pd.concat([vix_monthly, vvix_monthly, ff_monthly, mom_monthly], axis=1)
    # Columns: 'VIX', 'VVIX', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'.
    
    # Perform several linear regressions with various factors and save results in
    # results directory.
    print("Perform linear regressions according to the CAPM, 3FF model and 5FF model "
            "to try to explain the monthly long short portfolio returns...")
    regression_map = {
                    "CAPM": {  #folder in results where the below + stargazer latex will be saved.
                    "CAPM":                 ["Mkt-RF"],
                    "CAPM_MOM":             ["Mkt-RF", "MOM"],
                    "CAPM_MOM_VIX":         ["Mkt-RF", "MOM", "VIX"],
                    "CAPM_MOM_VIX_VVIX":    ["Mkt-RF", "MOM", "VIX", "VVIX"],
                    },
                    "3FF": {
                    "3FF":                  ["Mkt-RF", "SMB", "HML"],
                    "3FF_MOM":              ["Mkt-RF", "SMB", "HML", "MOM"],
                    "3FF_MOM_VIX":          ["Mkt-RF", "SMB", "HML", "MOM", "VIX"],
                    "3FF_MOM_VIX_VVIX":     ["Mkt-RF", "SMB", "HML", "MOM", "VIX", "VVIX"],
                    },
                    "5FF": {  
                    "5FF":                  ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
                    "5FF_MOM":              ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
                    "5FF_MOM_VIX":          ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "VIX"],
                    "5FF_MOM_VIX_VVIX":     ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "VIX", "VVIX"],
                    }
    }
    # Regress and save results of the regressions specified in regression_map on 
    # long_short portfolio return.The regression groups get summarized via the stargazer package.
    regress_factors(regression_map, factors_avail, long_short_pf_returns, path_results)
    print("Done.")
    print("All done!")


def feature_importance(args):
    # Ignore warnings as to not clutter the terminal here.
    import warnings
    warnings.filterwarnings('ignore')
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    start_time = time.time()
    # Get experiment path.
    logs_folder = Path.cwd()/"logs"
    matches = Path(logs_folder).rglob(args.expid) #Get folder in logs_folder that matches expid
    matches_list = list(matches)
    if not len(matches_list) == 1:
        raise ValueError(f"There exists none or more than 1 folder with "
                            f"experiment id {args.expid} in the {logs_folder.name} "
                            "directory!")
    exp_path = matches_list[0]
    # Get experiment args.
    args_exp = pd.read_json(exp_path/"args.json", typ="series")
    # Get original predictions.
    preds_orig = pd.read_csv(exp_path/"all_pred.csv", index_col=0) #shape [index, cols: ["id", "pred"]]
    # Get (year_idx, best_model_path) for each year into a list.
    # Note: year_idx starts at 0.
    yearidx_bestmodelpaths = get_yearidx_bestmodelpaths(exp_path, args_exp.model)
    # Data path.
    path_data = Path.cwd()/"data"
    # Load original feature_target dataframe to permute.
    # Note: pd.read_csv does not load 'date' column as datetime, in contrast to pd.read_parquet!
    orig_feature_target = load_data(path_data, args_exp.dataset)
    # List of features (columns of orig_feature_target).
    features_list = orig_feature_target.columns.tolist()
    features_list.remove("date")
    features_list.remove("option_ret")
    # How many sample permutation per feature?
    num_samples_per_feature = 20
    # Original long short monthly pf returns?
    path_portfolios = exp_path/"portfolios"
    if int(args_exp.label_fn[-1:]) == 5: # 5 class classification.
        # Note: pd.read_csv does not load 'date' column as datetime automatically, 
        # in contrast to pd.read_parquet!
        long_short_pf = pd.read_csv(path_portfolios/"long4short0.csv", parse_dates=["date"], index_col="date")
        long_short_pf_ret_orig = long_short_pf["option_ret"]
    else:
        raise NotImplementedError("Feature randomization only implemented for "
                                  "5 class classification for now")
    # Aggregate original optionreturns to sanity check long_short portfolio returns.
    test = orig_feature_target[["date", "option_ret"]].copy()
    check_orig = aggregate_newpred(preds_orig, test, args_exp)
    assert (abs(long_short_pf_ret_orig - check_orig) < 0.00001).all(), ("Loaded "
    "aggregated option returns have substantial differences to the check aggregation of the "
    "original option returns.")
    # ****

    results = loop_features(orig_feature_target=orig_feature_target, 
                         features_list=features_list, 
                         yearidx_bestmodelpaths=yearidx_bestmodelpaths,
                         preds_orig=preds_orig,
                         num_samples_per_feature=num_samples_per_feature,
                         args_exp=args_exp,
                         long_short_pf_ret_orig=long_short_pf_ret_orig
                         )

    # Sort results descending, once for balacc and once for meanofmean difference monthly pf returns.
    results_sorted_balacc = dict(sorted(results.items(), key= lambda item: item[1]["balacc"]["MeanDiffOrigPerm"], reverse=True))
    results_sorted_meanofmeandiffpf = dict(sorted(results.items(), key= lambda item: item[1]["pfret"]["MomDiff"], reverse=True))

    # Nested dict to DataFrame, full ols results.
    results_sorted_balacc_df = prepare_to_save(results_sorted_balacc, only_signif=False)
    results_sorted_meanofmeandiffpf_df = prepare_to_save(results_sorted_meanofmeandiffpf, only_signif=False)

    # Nested dict to DataFrame, only significance.
    results_sorted_balacc_df_onlysignif = prepare_to_save(results_sorted_balacc, only_signif=True)
    results_sorted_meanofmeandiffpf_df_onlysignif = prepare_to_save(results_sorted_meanofmeandiffpf, only_signif=True)

    #Save sorted results (once only significance, once with full ols results).
    path_importance = exp_path/"results"/"importance"
    path_importance.mkdir(exist_ok=True, parents=True)
    results_sorted_balacc_df.to_csv(path_importance/"balaccmeandiff_sorted_details.csv")
    results_sorted_balacc_df_onlysignif.to_csv(path_importance/"balaccmeandiff_sorted.csv")
    results_sorted_meanofmeandiffpf_df.to_csv(path_importance/"meanofmeandiffpf_sorted_details.csv")
    results_sorted_meanofmeandiffpf_df_onlysignif.to_csv(path_importance/"meanofmeandiffpf_sorted.csv")

    end_time = time.time()
    print("Finished. Feature importance completed in", end_time - start_time, "seconds.")


def loop_features(
                orig_feature_target: pd.DataFrame,
                features_list: list, 
                yearidx_bestmodelpaths: Tuple[int, Path],
                preds_orig: pd.DataFrame,
                num_samples_per_feature: int,
                args_exp: pd.Series,
                long_short_pf_ret_orig: pd.Series,
                ) -> pd.DataFrame:
    results = {}
    for feature in tqdm(features_list): # approx. 22sec per feature per sample.
        bal_acc_scores = {} #list
        ls_ret_avg_ols = {} #dict of list
        for _ in range(num_samples_per_feature):
            result = loop_years(orig_feature_target=orig_feature_target,
                                feature=feature,
                                yearidx_bestmodelpaths=yearidx_bestmodelpaths,
                                preds_orig=preds_orig,
                                args_exp=args_exp,
                                long_short_pf_ret_orig=long_short_pf_ret_orig,
                                )
            # Add to collection dict. Index must coincide with entry in list...
            bal_acc_scores.setdefault("Orig", []).append(result["diff_bal_acc_score"][0])
            bal_acc_scores.setdefault("Perm", []).append(result["diff_bal_acc_score"][1])
            bal_acc_scores.setdefault("DiffOrigPerm", []).append(result["diff_bal_acc_score"][2])
            bal_acc_scores.setdefault("Dummy", []).append(result["diff_bal_acc_score"][3])
            bal_acc_scores.setdefault("DiffOrigDummy", []).append(result["diff_bal_acc_score"][4])
            bal_acc_scores.setdefault("DiffPermDummy", []).append(result["diff_bal_acc_score"][5])

            # Add to collection dict.
            ls_ret_avg_ols.setdefault("MeanOrig", []).append(result["diff_long_short_ret"][0])
            ls_ret_avg_ols.setdefault("MeanNew", []).append(result["diff_long_short_ret"][1])
            ls_ret_avg_ols.setdefault("MeanDiff", []).append(result["diff_long_short_ret"][2])
            for key in result["diff_long_short_ret"][3]:
                ls_ret_avg_ols.setdefault(key, []).append(result["diff_long_short_ret"][3][key])
            
        # 1. Regress bal_acc_score mean differences on intercept.
        mean_orig = {"MeanOrig": np.mean(bal_acc_scores["Orig"])}
        mean_perm = {"MeanPerm": np.mean(bal_acc_scores["Perm"])}
        mean_dummy = {"MeanDummy": np.mean(bal_acc_scores["Dummy"])}
        # Get mean of differences, and its ols results in dictionaries.
        mean_diff_origperm, diff_bal_acc_ols_origperm = get_mean_ols_diff(bal_acc_scores, "DiffOrigPerm")
        mean_diff_origdummy, diff_bal_acc_ols_origdummy = get_mean_ols_diff(bal_acc_scores, "DiffOrigDummy")
        mean_diff_permdummy, diff_bal_acc_ols_permdummy = get_mean_ols_diff(bal_acc_scores, "DiffPermDummy")
        # Edit order of final dictionary. Reorder results.
        mean_bal_acc_ols = {**mean_orig, **mean_perm, **mean_diff_origperm, **diff_bal_acc_ols_origperm,
                            **mean_dummy, **mean_diff_origdummy, **diff_bal_acc_ols_origdummy,
                            **mean_diff_permdummy, **diff_bal_acc_ols_permdummy
                        } 
        # Is the mean equal (up to small error) to the * coefficient of the OLS
        # regression on the intercept?
        sanity_check_balacc_means(mean_bal_acc_ols)
        
        # 2. Average OLS results of monthly return differences.
        #NOTE: Here its a *mean of mean* differences! (Not as in 1. where its just a *mean* of differences...)
        mom_orig = {"MomOrig": np.mean(ls_ret_avg_ols["MeanOrig"])} #in order to move "Mean" to the beginning.
        mom_new = {"MomNew": np.mean(ls_ret_avg_ols["MeanNew"])} #in order to move "Mean" to the beginning.
        mom_diff = {"MomDiff": np.mean(ls_ret_avg_ols["MeanDiff"])} #in order to move "Mean" to the beginning.
        # Take mean of ols results and create final dict.
        avg_ols = {key: mean_str_add_stars(pd.concat(values)) for (key, values) in ls_ret_avg_ols.items() if not key.startswith("Mean")}
        mom_ls_ret_mean_ols = {**mom_orig, **mom_new, **mom_diff, **avg_ols}
        # Is the mean of mean equal (up to small error) to the *coefficient of 
        # the OLS regression on the intercept?
        sanity_check_mom_ls_means(mom_ls_ret_mean_ols)
        
        # Collect results for each feature. (Short key names for better csv. layout)
        results[f"{feature}"] = {"balacc": mean_bal_acc_ols, "pfret": mom_ls_ret_mean_ols}
    return results


def loop_years(
            orig_feature_target: pd.DataFrame,
            feature: str,
            yearidx_bestmodelpaths: list,
            preds_orig: pd.DataFrame,
            args_exp: pd.Series,
            long_short_pf_ret_orig: pd.Series,
            ) -> Dict[List[float], List[float]]: # (mean of monthly difference, t-value), (difference bal_acc score)
    
    # Permute feature.
    permuted_feature_target = orig_feature_target.copy()
    permuted_feature_target[feature] = np.random.permutation(permuted_feature_target[feature])
    
    # Which model was used in the experiment and has to be used also here?
    model_name = args_exp.model #string

    # Get predictions on permuted feature data for each year and concatenate.
    preds_perm_list = []
    preds_dummy_list = []
    for yearidx, bestmodelpath in yearidx_bestmodelpaths: #should be in ascending order.
        print(f"Loading trained model: '{model_name}' to predict on randomized feature from path: {bestmodelpath}")
        preds_perm_year, preds_dummy_year, y = pred_on_data(model_name, yearidx, bestmodelpath, 
                                                            permuted_feature_target, args_exp)
        preds_perm_list.append(preds_perm_year)
        preds_dummy_list.append(preds_dummy_year)
        # Actually makes some difference.
        import gc
        gc.collect()
    # Check true y vectors from different sources. Takes 'y' from the last 
    # for loop iteration -> should be the whole y of the data.
    check_y_classification(y, orig_feature_target, args_exp.label_fn)
    preds_perm = pd.concat(preds_perm_list).reset_index() #shape [index, [index, pred]]
    preds_dummy = pd.concat(preds_dummy_list).reset_index() #shape [index, [index, pred]]
    # Rename "index" to "id". (The checks look for a column 'id' since the 
    # original predictions first column is also named 'id'. # CRUCIAL for aggregate function later.
    preds_perm = preds_perm.rename(columns={"index": "id"})
    preds_dummy = preds_dummy.rename(columns={"index": "id"})
    # Get the relevant y vector corresponding to the test period (the predictions made accordingly).
    y_true = y[-len(preds_orig):] #last y_year is the y of the whole dataset.
    # Check whether length of y_true, preds_orig and preds_new are the same.
    if not (len(y_true) == len(preds_orig) == len(preds_perm) == len(preds_dummy)):
        raise ValueError("The true y values and the predictions do not coincide in length.")

    # 1. Output: Get difference of *test* balanced accuracy scores.
    bal_acc_orig = balanced_accuracy_score(y_true, preds_orig["pred"])
    bal_acc_dummy = balanced_accuracy_score(y_true, preds_dummy["pred"])
    bal_acc_diff_orig_dummy = bal_acc_orig - bal_acc_dummy # Expected to be positive (Better than random guessing)...
    bal_acc_perm = balanced_accuracy_score(y_true, preds_perm["pred"])
    bal_acc_diff_orig_perm = bal_acc_orig - bal_acc_perm # Expected to be positive! More positive -> more important feature.
    bal_acc_diff_perm_dummy = bal_acc_perm - bal_acc_dummy # Expected to be positive! Hopefully, still better than random guessing...

    # 2. Output: Get mean difference of monthly long-short portfolio returns.
    # Perform aggregation for new predictions.
    option_ret_to_agg = orig_feature_target[["date", "option_ret"]]
    ls_pf_ret_new = aggregate_newpred(preds_perm, option_ret_to_agg, args_exp) #aggregate preds after randomizing a feature.
    ls_pf_ret_diff = long_short_pf_ret_orig - ls_pf_ret_new.values
    ls_pf_ret_diff_mean = ls_pf_ret_diff.mean()
    ls_pf_ret_orig_mean = long_short_pf_ret_orig.mean()
    ls_pf_ret_perm_mean = ls_pf_ret_new.mean()
    #Regress differences with zero intercept model.
    # Is mean statistically significantly different from 0? -> Regress on intercept. 
    # # Standard OLS, HAC_maxlags, and HAC12 variants.
    diff_ols_results = regress_on_constant(ls_pf_ret_diff)
    # OLS coefficients and mean of difference should be equal (up to floating errors).
    sanity_check_ls_means(diff_ols_results, ls_pf_ret_diff_mean, ls_pf_ret_orig_mean, ls_pf_ret_perm_mean)

    results = {}
    results["diff_bal_acc_score"] = [bal_acc_orig, bal_acc_perm, bal_acc_diff_orig_perm, bal_acc_dummy, bal_acc_diff_orig_dummy, bal_acc_diff_perm_dummy]
    results["diff_long_short_ret"] = [ls_pf_ret_orig_mean, ls_pf_ret_perm_mean, ls_pf_ret_diff_mean, diff_ols_results]
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description=
        "Master Thesis Mathias Ruoss - Option Return Classification: "
        "Create portfolios from predictions."
    )
    subparsers = parser.add_subparsers()
    # 1. Aggregate returns via predictions to portfolios.
    parser_agg = subparsers.add_parser("agg")
    parser_agg.set_defaults(mode=aggregate)
    # 2.a) Performance evaluation of portfolios created with 'agg'.
    parser_perf = subparsers.add_parser("perf")
    parser_perf.set_defaults(mode=performance)
    # 2.b) Feature Importance (needs 'agg' to be done first).
    parser_perf = subparsers.add_parser("importance")
    parser_perf.set_defaults(mode=feature_importance)
    # Overhead Settings.
    cockpit = parser.add_argument_group("Overhead Configuration")
    cockpit.add_argument("expid", type=str, help="folder name of experiment, "
                        "given by time created")
    cockpit.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed) # Only relevant for feature importance permutation.
    
    args.mode(args) #aggreagte or performance
