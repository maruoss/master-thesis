from argparse import ArgumentParser
from pathlib import Path
import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from data.utils.convert_check import small_med_big_eq
from portfolio.helper import (
                            check_eoy, collect_preds, concat_and_save_preds, filter_idx, get_and_check_min_max_pred,
                            get_class_ignore_dates, get_yearidx_bestmodelpaths, regress_factors, save_performance_statistics, various_tests, 
                            weighted_means_by_column
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
    assert len(matches_list) == 1, "there exists none or more than 1 folder with given expid!"
    exp_dir = matches_list[0]
    # Move all predictions to 'predictions' folder with the expdir folder.
    print("Find all 'predictions{year}' files in the subfolders of the experiment "
    "and copy to 'predictions' folder...")
    collect_preds(exp_dir)
    print("Done.")
    #TODO: combine (ensemble) of multiple experiment predictions together? list of expids?
    # Read all prediction .csv and save as "all_pred.csv" in exp_dir.
    print("Read in all prediction .csv files as a dataframe and save as 'all_pred.csv'...")
    preds_concat_df = concat_and_save_preds(exp_dir)
    print("Done.")

    # Get path where datasets reside:
    print("Concat the dataframe with the respective option data...")
    datapath = Path.cwd()/"data"
    # Check whether small, med, big are all equal (whether we predicted on same 
    # test data ordering!)
    # Takes 11 secs., uncomment for final production code.
    # assert small_med_big_eq(datapath), ("Date and return columns NOT equal between "
    #                                 "small, medium and big datasets!")
    # print("Dates and option return columns from small, medium and big datasets are "
    #         "equal!")
    # Get small dataset (irrespective of small/medium/big used for train!).
    df_small = pd.read_parquet(datapath/"final_df_call_cao_small.parquet")
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
    assert len(matches_list) == 1, "there exists more than 1 folder with given expid!"
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
    monthly_rf = load_rf_monthly(path_data)
    # Filter months (rows) to align with pf_returns.
    monthly_rf = filter_idx(monthly_rf, pf_returns)

    # Subtract Rf from class portfolio returns to get excess returns.
    print("Subtract Riskfree rate *only from class* portfolios but *not* from the "
          "long short portfolio, as the riskfree rate cancels out for the long short "
          "portfolio...")
    class_pf_returns = pf_returns.columns.str.startswith("class")
    pf_returns.loc[:, class_pf_returns] = pf_returns.loc[:, class_pf_returns] - monthly_rf.values #.values needed to broadcast over columns.

    # Make 'results' subfolder.
    path_results = exp_path/"results"
    path_results.mkdir(exist_ok=True, parents=False)
    # Make 'performance' subfolder in 'results' folder.
    path_results_perf = exp_path/"results"/"performance"
    path_results_perf.mkdir(exist_ok=True, parents=False)

    # Save performance statistics in the 'performance' subfolder in exp_dir/'results'.
    print("Saving performance statistics...")
    save_performance_statistics(pf_returns, exp_path, path_data, path_results_perf)
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
                    "CAPM":                 ["Mkt-RF"],
                    "CAPM_MOM":             ["Mkt-RF", "MOM"],
                    "CAPM_MOM_VIX":         ["Mkt-RF", "MOM", "VIX"],
                    "CAPM_MOM_VIX_VVIX":    ["Mkt-RF", "MOM", "VIX", "VVIX"],
                    "3FF":                  ["Mkt-RF", "SMB", "HML"],
                    "3FF_MOM":              ["Mkt-RF", "SMB", "HML", "MOM"],
                    "3FF_MOM_VIX":          ["Mkt-RF", "SMB", "HML", "MOM", "VIX"],
                    "3FF_MOM_VIX_VVIX":     ["Mkt-RF", "SMB", "HML", "MOM", "VIX", "VVIX"],
                    "5FF":                  ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
                    "5FF_MOM":              ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
                    "5FF_MOM_VIX":          ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "VIX"],
                    "5FF_MOM_VIX_VVIX":     ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "VIX", "VVIX"],
    }
    # Regress the regressions specified in regression_map on long_short portfolio return.
    regress_factors(regression_map, factors_avail, long_short_pf_returns, path_results)
    print("Done.")

    print("All done!")


def filter_idx(df: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Filters indeces of 'df' to align with index of 'df_target'.
    
    Returns: Filtered DataFrame.
    """
    return df.loc[df.index[np.isin(df.index, df_target.index)]]


        Args:
            factors:        All independent variables concatenated in a Dataframe.
            y:              The dependent variable (long short monthly portfolio returns here).
            path_results:   The path where the results folder resides.
    
    """

    for regr_key in regression_map.keys():
        parent_path = path_results/regr_key
        parent_path.mkdir(exist_ok=True, parents=False)
        X = factors_avail.loc[:, regression_map[regr_key]]
        # Add constant for intercept.
        X = sm.add_constant(X)

        # 1a. Regression. No HAC Standard Errors.
        ols = sm.OLS(y, X) #long_short_return regressed on X.
        ols_result = ols.fit()
        regr_type = "Standard"
        save_ols_results(ols_result, parent_path, regr_type)

        # 1b. Regression. With HAC Standard Errors. Lags of Greene (2012): L = T**(1/4).
        max_lags = int(len(X)**(1/4))
        ols_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
        regr_type = f"HAC_{max_lags}"
        save_ols_results(ols_result, parent_path, regr_type)

        # 1c. Regression. With HAC Standard Errors. Lag after Bali (2021) = 12.
        ols_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        regr_type = "HAC_12"
        save_ols_results(ols_result, parent_path, regr_type)


def save_ols_results(ols_result, parent_path: Path, fileprefix: str) -> None:
    """Saves results from statsmodels.OLS to .txt and .csv files in a separate 
    folder in the 'parent_path' path.

        Args:
            ols_result:             Object from ols.fit().summary()
            ols_result2:            Object from ols.fit().summary2()
            parent_path (Path):     Path to the results folder.
            foldername (str):       Name of the folder to be created at results/foldername.
    
    """
    # Create separate folder to save the regression results in.
    # reg_folder = parent_path/foldername
    # reg_folder.mkdir(exist_ok=True, parents=False)
    # 3 LaTeX files.
    with (parent_path/f"{fileprefix}_result_latex1.txt").open("w") as text_file:
        text_file.write("OLS Summary (For Loop):\n\n")
        for table in ols_result.summary(alpha=0.05).tables:
            text_file.write(table.as_latex_tabular())
    with (parent_path/f"{fileprefix}_result_latex2.txt").open("w") as text_file:
        text_file.write("OLS Summary as LaTeX:\n\n")
        text_file.write(ols_result.summary(alpha=0.05).as_latex())
    with (parent_path/f"{fileprefix}_result_latex3.txt").open("w") as text_file:
        text_file.write("OLS Summary2 as LaTeX:\n\n")
        text_file.write(ols_result.summary2(alpha=0.05).as_latex())
    # 1 .txt file.
    with (parent_path/f"{fileprefix}_result.txt").open("w") as text_file:
        text_file.write(ols_result.summary().as_text())
    # 1 .csv file.
    with (parent_path/f"{fileprefix}_result.csv").open("w") as text_file:
        text_file.write(ols_result.summary().as_csv())


if __name__ == "__main__":
    parser = ArgumentParser(description=
        "Master Thesis Mathias Ruoss - Option Return Classification: "
        "Create portfolios from predictions."
    )
    subparsers = parser.add_subparsers()

    parser_agg = subparsers.add_parser("agg")
    parser_agg.set_defaults(mode=aggregate)

    parser_perf = subparsers.add_parser("perf")
    parser_perf.set_defaults(mode=performance)
    
    cockpit = parser.add_argument_group("Overhead Configuration")
    cockpit.add_argument("expid", type=str, help="folder name of experiment, "
                        "given by time created")

    args = parser.parse_args()
    
    args.mode(args) #aggreagte or performance
