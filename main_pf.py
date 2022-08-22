from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
import pdb
from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
import yfinance as yf
import dataframe_image as dfi
import quantstats as qs

from arguments import load_args
from data.utils.convert_check import small_med_big_eq
from portfolio.helper import check_eoy, collect_preds, concat_and_save_preds, get_and_check_min_max_pred, various_tests, weighted_avg, weighted_means_by_column, weighted_means_by_column2
from utils.preprocess import YearMonthEndIndeces


def aggregate(args):
    # Get experiment folder path 'exp_dir'.
    logs_folder = Path.cwd()/"logs"
    matches = Path(logs_folder).rglob(args.expid) #Get folder in logs_folder that matches expid
    matches_list = list(matches)
    assert len(matches_list) == 1, "there exists more than 1 folder with given expid!"
    exp_dir = matches_list[0]
    # Move all predictions to 'predictions' folder with the expdir folder.
    collect_preds(exp_dir)
    #TODO: combine (ensemble) of multiple experiment predictions together? list of expids?
    # Read all prediction .csv and save as "all_pred.csv" in exp_dir.
    preds_concat_df = concat_and_save_preds(exp_dir)

    # Get path where datasets reside:
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

    # Create weight column for lowest and highest predicted classes (for L-S portfolio)
    max_pred, min_pred = get_and_check_min_max_pred(concat_df, args_exp["label_fn"])
    classes = sorted(concat_df["pred"].unique(), reverse=False) #ascending order
    assert max_pred == classes[-1] and min_pred == classes[0], ("min and max preds "
                                                                "are not attained")
    # Delete below in production---
    class_weight_map = {}
    for c in range(len(classes)):
        if c == min_pred:
            class_weight_map[c] = -1
        elif c == max_pred:
            class_weight_map[c] = 1
        else:
            class_weight_map[c] = 0
    # Apply class map.
    test  = concat_df["pred"].map(class_weight_map)
    # ---
    # 1.5x faster than pd.map...
    condlist = [concat_df["pred"] == min_pred, concat_df["pred"] == max_pred]
    choicelist = [-1, 1]
    no_alloc_value = 0
    concat_df["if_long_short"] = np.select(condlist, choicelist, no_alloc_value)
    # Delete below in production...
    assert (test == concat_df["if_long_short"]).all(), "Two methods to create weightings do not yield same result."
    # ---

    # Create separate weight column for each class in concat_df.
    for c in classes:
        condlist = [concat_df["pred"] == c]
        choicelist = [1]
        no_alloc_value = 0
        concat_df[f"weights_{c}"] = np.select(condlist, choicelist, no_alloc_value)

    # Only calculate weighted average for numerical columns.
    col_list = [val for val in concat_df.columns.tolist() if "weight" not in val 
                and "date" not in val]

    # Collect all portfolios in a dictionary with key 'class0', 'class1', etc.
    agg_dict = {}
    for c in classes:
        agg_df = concat_df.groupby("date").aggregate(weighted_means_by_column, col_list, f"weights_{c}")
        agg_dict[f"class{c}"] = agg_df

    # Perform various tests to check our calculations.
    various_tests(concat_df, col_list, classes, agg_dict)

    # Save all aggregated dataframes per class to 'portfolios' subfolder within the 
    # experiment directory 'exp_dir'.
    pf_dir = exp_dir/"portfolios"
    try: #if portfolios folder doesnt exist...
        pf_dir.mkdir(exist_ok=False, parents=False) # raise Error if parents are missing.
        for c, df in agg_dict.items():
            df.to_csv(exp_dir/pf_dir/f"agg_df_{c}.csv")
    except FileExistsError: # portfolios folder already exists, do nothing.
        print("Directory 'portfolios' already exists. Will leave as is and continue with the code.")

    # # Get geometric mean of Long-Short PF
    # long_short_monthly = (agg_dict[f"class{classes[-1]}"] - agg_dict[f"class{classes[0]}"])
    # cols_to_keep = [col for col in long_short_monthly.columns.tolist() if "weight" not in col]
    # long_short_monthly = long_short_monthly[cols_to_keep]

    # TODO: create LaTeX output for portfolio mean, sd, sharpe ratio as in Bali (2021)?

def performance(args):
    sns.set_style("whitegrid") # style must be one of white, dark, whitegrid, darkgrid, ticks
    sns.set_context('paper', font_scale=2.0) # set fontsize scaling for labels, axis, legend, automatically moves legend

    # Get experiment folder path 'exp_dir'.
    logs_folder = Path.cwd()/"logs"
    matches = Path(logs_folder).rglob(args.expid) #Get folder in logs_folder that matches expid
    matches_list = list(matches)
    assert len(matches_list) == 1, "there exists more than 1 folder with given expid!"
    exp_path = matches_list[0]

    # Read aggregated portfolios.
    path_portfolios = exp_path/"portfolios"
    dfs = []
    for file in Path.iterdir(path_portfolios):
        try:
            df = pd.read_csv(path_portfolios/file, parse_dates=["date"], index_col="date")
        except PermissionError as err:
            raise PermissionError("The 'portfolios' subfolder must not contain directories.") from err
            # from err necessary for chaining
        dfs.append(df["option_ret"].rename(file.name[7:-4])) #rename Series to 'class0', 'class1', etc.
    dfs = pd.concat(dfs, axis=1) # Series names -> column names
    # Capitalize 'date' index name for plot axis label.
    dfs.index = dfs.index.rename("Date")

    # Plot equity line and save to .png.
    dfs_cumprod = (1. + dfs).cumprod()
    dfs_cumprod.plot(figsize=(15, 10), alpha=1.0, linewidth=2.5, ylabel="Portfolio Value",
                    title="Cumulative Return of Classification Portfolios")
    plt.tight_layout() # remove whitespace around plot

    # Make 'results' subfolder.
    path_results = exp_path/"results"
    path_results.mkdir(exist_ok=True, parents=False)
    plt.savefig(path_results/"plot.png")

    # Collect performance statistics. 
    # RISKFREE rate assumed zero here. Not implemented in the formulas.
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
    for columname in dfs.columns:
        col_summary = qs.stats.drawdown_details(qs.stats.to_drawdown_series(dfs))[columname].sort_values(by="max drawdown", ascending=True)
        # Divide by 100 because qs.stats.drawdown_details provides maxdd in percent.
        assert col_summary["max drawdown"].iloc[0] / 100 == maxdd[columname], "Max Drawdown of qs is not equal our manually calculated Max. DD"
        maxdd_days.append(col_summary["days"].iloc[0])
    maxdd_days = pd.Series(maxdd_days, index=dfs.columns, name="Max DD days")
    maxdd_days_str = maxdd_days.apply(lambda x: f'{x: .0f}')

    # Calmar Ratio.
    calmar = cagr / maxdd.abs()
    calmar_str = calmar.apply(lambda x: f'{x: .3f}').rename("Calmar Ratio")

    # Skewness/ Kurtosis.
    skew = dfs.skew()
    skew_str = skew.apply(lambda x: f'{x: .3f}').rename("Skewness")
    kurt = dfs.kurtosis()
    kurt_str = kurt.apply(lambda x: f'{x: .3f}').rename("Kurtosis")

    # cumr = qs.stats.comp(dfs).rename("Cumulative Return")
    # cumr = cumr.apply(lambda x: f'{x: .2%}')
    # cagr = qs.stats.cagr(dfs).rename("CAGR") # assumes rf = 0
    # cagr = cagr.apply(lambda x: f'{x: .2%}')
    # vol = qs.stats.volatility(dfs, periods=periods).rename("Volatility (ann.)") # daily to annualized vol.
    # vol = vol.apply(lambda x: f'{x: .2%}')
    # sharpe = qs.stats.sharpe(dfs).rename("Sharpe Ratio")
    # sharpe = sharpe.apply(lambda x: f'{x: .2f}')
    # maxdd = qs.stats.max_drawdown(dfs).rename("Max Drawdown")
    # maxdd = maxdd.apply(lambda x: f'{x: .2%}')
    # dd_days = []
    # for columname in dfs.columns:
    #     dd_days.append(qs.stats.drawdown_details(qs.stats.to_drawdown_series(dfs))[columname].sort_values(by="max drawdown", ascending=True)["days"].iloc[0])
    # dd_days = pd.Series(dd_days, index=dfs.columns, name="Max DD days")
    # dd_days = dd_days.apply(lambda x: f'{x: .0f}')
    # calmar = qs.stats.calmar(dfs).rename("Calmar Ratio")
    # calmar = calmar.apply(lambda x: f'{x: .2f}')
    # skew = qs.stats.skew(dfs).rename("Skewness")
    # skew = skew.apply(lambda x: f'{x: .2f}')
    # kurt = qs.stats.kurtosis(dfs).rename("Kurtosis")
    # kurt = kurt.apply(lambda x: f'{x: .2f}')

    # # Calculate alpha, betas.
    # X = np.array([np.ones_like(dfs.DJIA), dfs.DJIA]).T
    # alphabeta = np.linalg.inv(X.T@X)@X.T@dfs
    # alpha = alphabeta.iloc[0, :] * periods
    # beta = alphabeta.iloc[1, :]
    # alpha = alpha.rename("Alpha")
    # beta = beta.rename("Beta")
    # alpha = alpha.apply(lambda x: f'{x: .2f}')
    # beta = beta.apply(lambda x: f'{x: .2f}')

    #append more stats first here...

    # perfstats += [cumr, cagr, vol, sharpe, maxdd, dd_days, calmar, skew, kurt, alpha, beta] # then here.
    perfstats += [cumr_str, cagr_str, mean_ann_str, vol_ann_str, sharpe_ann_str, maxdd_str, maxdd_days_str, calmar_str, skew_str, kurt_str, alphas_str, betas_str]

    # Save results to 'results' subfolder.
    perfstats = pd.concat(perfstats, axis=1)
    perfstats.to_csv(path_results/"perfstats.csv")
    # dfi only accepts strings as paths:
    dfi.export(perfstats, str(path_results/"perfstats.png"))
    # dfi.export(perfstats, os.path.join(path_results, "perfstats.png")) #Alternative

    # Export latex code for table.
    with (path_results/"latex.txt").open("w") as text_file:
        text_file.write(perfstats.style.to_latex())
        text_file.write("\n")
        text_file.write("% Same table transposed:\n")
        text_file.write("\n")
        text_file.write(perfstats.T.style.to_latex())
    # Alternative via Pathlib (but have cannot chain inputs, I believe?).
    # latex_file = (path_results/"latex2.txt")
    # latex_file.write_text(perfstats.style.to_latex())


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
    
    args.mode(args) #TODO: run or aggregate?
