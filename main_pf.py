from argparse import ArgumentParser
from pathlib import Path
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
import quantstats as qs

from data.utils.convert_check import small_med_big_eq
from portfolio.helper import check_eoy, collect_preds, concat_and_save_preds, export_dfi, get_and_check_min_max_pred, get_class_ignore_dates, various_tests, weighted_means_by_column
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
    print("Done!")
    #TODO: combine (ensemble) of multiple experiment predictions together? list of expids?
    # Read all prediction .csv and save as "all_pred.csv" in exp_dir.
    print("Read in all prediction .csv files as a dataframe and save as 'all_pred.csv'...")
    preds_concat_df = concat_and_save_preds(exp_dir)
    print("Done!")

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
    df_small = pd.read_parquet(datapath/"final_df_small.parquet")
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
    print("Done!")

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
    print("Done!")
    # Aggregate and collect all portfolios in a dictionary with key 'class0', 'class1', etc.
    print("Aggregate for each class and collect the dataframes...")
    agg_dict = {}
    for c in classes:
        agg_df = concat_df.groupby("date").aggregate(weighted_means_by_column, col_list, f"weights_{c}")
        agg_dict[f"class{c}"] = agg_df
    print("Done!")
    
    print("Which classes were not predicted at all in a respective month?...")
    # For each class print out months where no prediction was allocated for that class, 
    # and save these indeces for short and long class to later ignore the returns of 
    # these months.
    class_ignore = get_class_ignore_dates(concat_df, classes) #returns dict
    print("Done!")
    
    # Perform various tests to check our calculations.
    test_concat = concat_df.copy()
    test_agg_dict = agg_dict.copy()
    print("Sanity test the aggregated results...")
    various_tests(agg_dict, concat_df, col_list, classes, class_ignore)
    print("Done!")
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
    print("Done!")

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
    print("Done!")
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

    # Read aggregated portfolios.
    print("Reading aggregated portfolios from 'portfolios' folder...")
    path_portfolios = exp_path/"portfolios"
    dfs = []
    for file in Path.iterdir(path_portfolios):
        try:
            df = pd.read_csv(path_portfolios/file, parse_dates=["date"], index_col="date")
        except PermissionError as err:
            raise PermissionError("The 'portfolios' subfolder must not contain directories.") from err
            # from err necessary for chaining
        dfs.append(df["option_ret"].rename(file.name[:-4])) #rename Series to filename from portfolios.
    # Sort list in descending order first -> class0, class1, ..., etc.
    dfs = sorted(dfs, key=lambda x: x.name)
    dfs = pd.concat(dfs, axis=1) # Series names -> column names
    # Capitalize 'date' index name for plot axis label.
    dfs.index = dfs.index.rename("Date")
    print("Done!")

    # Plot equity line and save to .png.
    print("Plotting equity lines...")
    dfs_cumprod = (1. + dfs).cumprod()
    dfs_cumprod.plot(figsize=(15, 10), alpha=1.0, linewidth=2.5, ylabel="Portfolio Value",
                    title="Cumulative Return of Classification Portfolios")
    plt.tight_layout() # remove whitespace around plot
    # Make 'results' subfolder.
    path_results = exp_path/"results"
    path_results.mkdir(exist_ok=True, parents=False)
    plt.savefig(path_results/"plot.png")
    print("Done!")

    # Collect performance statistics. 
    # RISKFREE rate assumed zero here. Not implemented in the formulas.
    print("Calculating portfolio performance statistics...")
    perfstats = []
    periods = 12 # we have monthly (excess) returns (as rf assumed 0.)

    # Convert decimals to percentages of a pandas series.
    def to_pct(x):
        """Transforms decimal to % string"""
        return f'{x:.2%}'

    # Cumulative Returns.
    cumr = (dfs+1).prod() - 1
    cumr_str = cumr.apply(to_pct).rename("Cum. Return")

    # Annualized arithmetic mean.
    mean_ann = dfs.mean() * periods
    mean_ann_str = mean_ann.apply(to_pct).rename("Mean (ann.)")

    # Cagr from qs.stats.cagr is wrong (subtracts 1 from lenght of total months, bc it assumes its days)
    cagr = (dfs+1).prod() ** (periods/len(dfs)) - 1
    cagr_str = cagr.apply(to_pct).rename("CAGR")

    # Annualized volatility.
    vol_ann = dfs.std(ddof=1) * np.sqrt(periods)
    vol_ann_str = vol_ann.apply(to_pct).rename("Volatility (ann.)")

    # Annualized Sharpe Ratio.
    sharpe_ann = mean_ann / vol_ann
    sharpe_ann_str = sharpe_ann.apply(lambda x: f'{x: .3f}').rename("Sharpe Ratio (ann.)")

    # Geometric annualized Sharpe Ratio
    sharpe_ann_geom = cagr / vol_ann
    sharpe_ann_geom_str = sharpe_ann_geom.apply(lambda x: f'{x: .3f}').rename("Geom. Sharpe Ratio (ann.)")

    # Calculate alpha and beta w.r.t. SP500 Total Return index (data from yfinance, see data/utils/sp500.py)
    # Read data from /data.
    path_data = Path.cwd()/"data"
    sp500_monthend = pd.read_csv(path_data/"sp500TR_prices.csv", parse_dates=["Date"], index_col="Date")
    # Select only relevant eom prices, and one month further back -> to calculate returns.
    sp500_ret = sp500_monthend.iloc[-len(dfs.index) - 1:]["Adj Close"].pct_change()
    # Remove first NaN row.
    sp500_ret = sp500_ret.iloc[1:]
    # Perform linear regression.
    X = np.array([np.ones_like(sp500_ret), sp500_ret]).T #bring to shape [1, sp500]
    alphabetas = np.linalg.inv(X.T@X)@X.T@dfs #regress dfs (y) on [1, sp500] (X)
    alphas = alphabetas.iloc[0, :] * periods #annualized alpha
    betas = alphabetas.iloc[1, :]
    alphas_str = alphas.apply(lambda x: f'{x: .3f}').rename("Alpha (ann.)")
    betas_str = betas.apply(lambda x: f'{x: .3f}').rename("Beta")

    # Max Drawdown.
    dfsdd = dfs.copy()
    # Insert [0, 0, ..., 0] as first prices, to calculate MaxDD correctly.
    dfsdd.loc[dfsdd.index[0] - MonthEnd(1)] = [0] * dfsdd.shape[1]
    dfsdd = dfsdd.sort_index()
    prices = (1 + dfsdd).cumprod()
    maxdd = (prices / prices.expanding().max()).min() - 1 #formula from quantstats
    maxdd_str = maxdd.apply(to_pct).rename("Max Drawdown")

    #Max Drawdown Days.
    maxdd_days = []
    for columname in dfsdd.columns:
        # Important to take 'dfsdd' here and not 'dfs'...
        col_summary = qs.stats.drawdown_details(qs.stats.to_drawdown_series(dfsdd))[columname].sort_values(by="max drawdown", ascending=True)
        # Divide by 100 because qs.stats.drawdown_details provides maxdd in percent.
        qs_dd = col_summary["max drawdown"].iloc[0] / 100
        assert (qs_dd - maxdd[columname]) < 0.0001, (f"Max Drawdown of column {columname}: {qs_dd} calculated by quantstats"
        f"is not equal our manually calculated Max. DD {maxdd[columname]}")

        maxdd_days.append(col_summary["days"].iloc[0])
    maxdd_days = pd.Series(maxdd_days, index=dfsdd.columns, name="Max DD days")
    maxdd_days_str = maxdd_days.apply(lambda x: f'{x: .0f}')

    # Calmar Ratio.
    calmar = cagr / maxdd.abs()
    calmar_str = calmar.apply(lambda x: f'{x: .3f}').rename("Calmar Ratio")

    # Skewness/ Kurtosis.
    skew = dfs.skew()
    skew_str = skew.apply(lambda x: f'{x: .3f}').rename("Skewness")
    kurt = dfs.kurtosis()
    kurt_str = kurt.apply(lambda x: f'{x: .3f}').rename("Kurtosis")

    # perfstats += [cumr, cagr, vol, sharpe, maxdd, dd_days, calmar, skew, kurt, alpha, beta] # then here.
    perfstats += [cumr_str, cagr_str, mean_ann_str, vol_ann_str, sharpe_ann_str, sharpe_ann_geom_str, maxdd_str, maxdd_days_str, calmar_str, skew_str, kurt_str, alphas_str, betas_str]

    print("Save performance statistics to .csv, .png and .txt...")
    # Save results to 'results' subfolder.
    perfstats = pd.concat(perfstats, axis=1)
    perfstats.to_csv(path_results/"perfstats.csv")
    # dfi only accepts strings as paths:
    export_dfi(perfstats, str(path_results/"perfstats.png"))

    # Export latex code for table.
    with (path_results/"latex.txt").open("w") as text_file:
        text_file.write(perfstats.style.to_latex())
        text_file.write("\n")
        text_file.write("% Same table transposed:\n")
        text_file.write("\n")
        text_file.write(perfstats.T.style.to_latex())
    print("Done!")
    print("All done!")


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
