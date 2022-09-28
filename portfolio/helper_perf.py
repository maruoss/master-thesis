from glob import escape
from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
import dataframe_image as dfi
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
import quantstats as qs
import statsmodels.api as sm
from pandas.api.types import is_integer_dtype 

from portfolio.helper_ols import get_save_alphabeta_significance, get_save_mean_significance
from portfolio.load_files import load_mkt_excess_ret_monthly


def collect_preds(exp_dir: Path) -> None:
    """Copies all predictions????.csv to a 'predictions' folder within the
    experiment_directory."""
    preds_dir = exp_dir/"predictions"
    preds_dir.mkdir(exist_ok=True, parents=True)
    # For all objects in exp_dir.
    for dir in exp_dir.iterdir():
        if dir.is_dir() and dir.name != "predictions":
            # See https://docs.python.org/3/library/fnmatch.html#module-fnmatch
            # for filename pattern matching below.
            for file in dir.glob("prediction[1,2]???.csv"): #[1,2]??? for years 1995,..,2000,..
                # If files do not exist in 'predictions' folder yet
                if not (preds_dir/(file.name)).is_file():
                    print(f"Copy file: '{file.relative_to(Path.cwd())}'"
                        f"to '{preds_dir.relative_to(Path.cwd())}'")
                    try:
                        shutil.copy2(file, preds_dir)
                    except shutil.SameFileError:
                        print("Source and Destination are the same file...")
                else:
                    print(f"File {file.name} already exists in '{preds_dir.name}' folder. "
                            "Will not touch it.")


def concat_and_save_preds(exp_dir: Path) -> pd.DataFrame:
    """Read prediction????.csv files from the 'predictions' folder in the experiment
    directory and return the concatenated pandas dataframe.
    
    Also, make sure years of 'prediction????.csv' are read in ascending order
    and consecutively, i.e. 2009, 2010, 2011, ... and not 2009, 2011, ... .
    """
    preds_dir = exp_dir/"predictions"
    preds = []
    prev_year = 0 # to make sure that files are read from lowest years to top years.
    for idx, file in enumerate(sorted(preds_dir.glob("*.csv"), reverse=False)): 
        # MUST BE reverse=False <=> ascending!, so that years are read in order ->2003->2004, etc.
        if not idx: # For first year (equiv. to: if idx == 0).
            year = int(file.stem[-4:]) 
            assert year > prev_year, "ERROR: year is not a positive integer"
            prev_year = year
        else: # For remaining years: must be consecutive, i.e. 2010, 2011, etc.
            year = int(file.stem[-4:]) 
            assert year == prev_year + 1, "ERROR: year is not succeeding previous year"
            prev_year = year
        pred_df = pd.read_csv(file)
        if not is_integer_dtype(pred_df["pred"]): #if classes are still floats (xgb case, fixed now.)
            print("Predictions are not integers, convert to int...")
            pred_df["pred"] = pred_df["pred"].astype(int)
        preds.append(pred_df)
    # Return concatenated dataframe.
    preds_concat_df = pd.concat(preds).reset_index(drop=True)
    preds_concat_df.to_csv(exp_dir/"all_pred.csv")
    return preds_concat_df


# Checks whether rows with id of 0 correspond to start of new years.
def check_eoy(concat_df: pd.DataFrame, eoy_indeces: np.ndarray):
    """Checks whether start of year (eoy indeces) rows correspond to id 0 in 
    concatenated predictions.
    
    """
    id_eq_zero = np.where(concat_df.loc[:, "id"] == 0)[0] #np.where returns tuple
    for idx_idx, idx in enumerate(id_eq_zero):
        if concat_df.iloc[idx]["index"] != eoy_indeces[idx_idx]:
            return False
    return True


def get_and_check_min_max_pred(concat_df: pd.DataFrame, labelfn_exp: str):
    """Checks whether the predictions contain at least one of the smallest and
    at least one of the largest class in each month (so that we can form Long-
    Short Portfolios.)

    Arguments: 
        concat_df:      Dataframe with option returns and the direction prediction. 
        labelfn_exp:    The label_fn of the experiment. Should be a string 'binary' 
                        or 'multi{number of classes}'.
    Returns:
        max_real:       Max realized prediction over all the data.
        min_real:       Min realized prediction over all the data.
        """
    classes = sorted(concat_df["pred"].unique(), reverse=False) #ascending order
    # Min pred value theoretically.
    min_theor = 0
    if labelfn_exp=="binary":
        max_theor = 1 
    else: #multi3, multi5, multi10 -> take (3, 5, 10) - 1
        max_theor = int(labelfn_exp[5:]) - 1 # 3 classes -> 0, 1, 2
    assert len(classes) == max_theor + 1, "At least one class is not predicted at all."
    assert classes[0] == min_theor and classes[-1] == max_theor, "List 'classes' is not sorted in ascending order."
    # Min pred value realized per month.
    min_real_series = concat_df.groupby("date")["pred"].min()
    min_real = min_real_series.min()
    # print("Min prediction realized is:", min_real)
    assert min_theor == min_real, (
        "Not a single month has the prediction of the theoretical minimum class.")
    months_no_min = min_real_series[min_real_series != min_real].count()
    print(f"Number of months where lowest class {min_real} is not predicted:", 
            months_no_min, "out of", f"{len(min_real_series)}.")
    # Max pred value realized per month.
    max_real_series = concat_df.groupby("date")["pred"].max()
    max_real = max_real_series.max()
    # max_real_series[max_real_series != max_real].index.strftime("%Y-%m-%d").to_list()
    assert max_theor == max_real, (
        "Not a single month has the prediction of the theoretical maximum class.")
    months_no_max = max_real_series[max_real_series != max_real].count()
    print(f"Number of months where largest class {max_real} is not predicted:", 
            months_no_max, "out of", f"{len(max_real_series)}.")
    return max_real, min_real, classes


def aggregate_threshold(concat_df, classes, col_list, agg_func, min_pred: int):
    agg_dict = {}
    for c in classes:
        agg_df = concat_df.groupby("date").aggregate(agg_func, col_list, f"weights_{c}")
        count = concat_df.groupby("date")[f"weights_{c}"].aggregate("sum")
        # length = concat_df.groupby("date")[f"weights_{c}"].aggregate(lambda x: len(x))
        # length2 = concat_df.groupby("date")[f"weights_{c}"].aggregate("count")
        # pct = concat_df.groupby("date")[f"weights_{c}"].aggregate(lambda x: sum(x) / len(x))
        # SET INVESTMENT TO ZERO BELOW CERTAIN CUTOFF OF # PREDICTIONS
        agg_df = agg_df.where(count > min_pred, 0) 
        agg_dict[f"class{c}"] = agg_df
    return agg_dict


def get_long_short_df(agg_dict, classes, class_ignore, shortclass, longclass):
    # Subtract short from long portfolio.
    long_df = agg_dict[f"class{longclass}"].copy() #deep copy to not change original agg_dict
    short_df = agg_dict[f"class{shortclass}"].copy() #deep copy to not change original agg_dict
    months_no_inv = class_ignore[f"class{longclass}"].union(class_ignore[f"class{shortclass}"]) #union of months to set to 0.
    long_df.loc[months_no_inv, :] = 0
    short_df.loc[months_no_inv, :] = 0
    long_short_df = long_df - short_df #months that are 0 in both dfs stay 0 everywhere.
    assert ((long_short_df.drop(months_no_inv)["pred"] == (longclass - shortclass)).all() and #'pred' should be long_class - short_class
        (long_short_df.drop(months_no_inv)["if_long_short"] == 2).all()) #'if_long_short' should be 2 (1 - (-1) = 2)
    print(f"Summary: In {len(months_no_inv)} months or in {len(months_no_inv)/len(long_short_df):.2%} "
        f"of the total {len(long_short_df)} months there has not been made a long-short investment.")
    return long_short_df


def various_tests(agg_dict: dict, concat_df: pd.DataFrame, col_list: list, classes: list, 
                    class_ignore: dict, min_pred: int, longclass: int):
    """Perform various sanity checks on our monthly aggregated results in agg_dict."""
    # Test1: Compare agg_dict with agg_dict2, calculated via 'weighted_avg' function 
    # and not via 'np.average'. They should yield the same (up to small precision).
    agg_dict2 = aggregate_threshold(concat_df, classes, col_list, 
                                    weighted_means_by_column2, min_pred)
    for key in agg_dict.keys():
        pd.testing.assert_frame_equal(agg_dict[key], agg_dict2[key])
    print("Test1: Successful! Weighted_avg function seems to yield the same as np.average.")

    # COPY CRUCIAL HERE! Otherwise, input df will be altered...
    agg_dict_copy = agg_dict.copy() #copy because we drop class_ignore months for each class.
    concat_df_copy = concat_df.copy()
    # Drop 'class_ignore' rows:
    for c in classes:
        agg_dict_copy[f"class{c}"] = agg_dict_copy[f"class{c}"].drop(class_ignore[f"class{c}"])
    # Test2: Check whether first and last month aggregation yield same as first 
    # and last entries of agg_dict_copy for each class.
    for c in classes:
        concat_df_copy_c = concat_df_copy[~concat_df_copy["date"].isin(class_ignore[f"class{c}"])]
        first_month = concat_df_copy_c.loc[concat_df_copy_c["date"] == concat_df_copy_c["date"].iloc[0]]
        last_month = concat_df_copy_c.loc[concat_df_copy_c["date"] == concat_df_copy_c["date"].iloc[-1]]
        for k in col_list:
            assert np.average(first_month[k], weights=first_month[f"weights_{c}"]) == agg_dict_copy[f"class{c}"].iloc[0][k]
            assert np.average(last_month[k], weights=last_month[f"weights_{c}"]) == agg_dict_copy[f"class{c}"].iloc[-1][k]
            assert (weighted_avg(first_month, k, f"weights_{c}") - agg_dict_copy[f"class{c}"].iloc[0][k]) < 0.0001
            assert (weighted_avg(last_month, k, f"weights_{c}") - agg_dict_copy[f"class{c}"].iloc[-1][k]) < 0.0001
    print("Test2: Successful! First and last month individual aggregation (of non-to-ignore months) yield the same "
         "as first and last entries of the aggregated dataframe for the respective class.")

    # Test3: If "pred" column in aggregated df's corresponds to class in each row (month).
    for c in classes:
        assert (agg_dict_copy[f"class{c}"]["pred"] == c).all(), "Aggregated 'pred' is not equal to the class in at least one month."
    print("Test3: Successful! Aggregated 'pred' column is equal to the class in each month.")
    # Test4: If short and low portfolios are aggregated correctly.
    assert ((agg_dict_copy[f"class{classes[0]}"]["if_long_short"] == -1).all() and
            (agg_dict_copy[f"class{longclass}"]["if_long_short"] == 1).all()), ("Long "
            "or short portfolio aggregation does not yield 1 or -1 in 'if_long_short' column.")
    print("Test4: Successful! Both the lowest class and the highest class corrrespond "
        "to -1 and 1 in the column 'if_long_short', respectively.")
    # Test5: Check if one-hot encoding columns correspond to 'preds' and 'if_long_short'.
    for c in classes:
        for k in classes:
            if c == k:
                assert (agg_dict_copy[f"class{c}"][f"weights_{k}"] == 1).all()
                assert (agg_dict_copy[f"class{c}"]["pred"] == k).all()
                if c==classes[0]:
                    assert (agg_dict_copy[f"class{c}"]["if_long_short"] == -1).all()
                elif c==longclass:
                    assert (agg_dict_copy[f"class{c}"]["if_long_short"] == 1).all()
                else:
                    assert (agg_dict_copy[f"class{c}"]["pred"] == k).all()
            else:
                assert (agg_dict_copy[f"class{c}"][f"weights_{k}"] == 0).all()
    print("Test5: Successful! Check whether one-hot encoding columns make sense "
        "with the columns 'preds' and 'if_long_short'.")


# Weighted average functions used to aggreagte portfolios. We use np.average.
def weighted_means_by_column(x, cols, w):
    """ This takes a DataFrame and averages each data column (cols)
        while weighting observations by column w.
    """
    try:
        return pd.Series([np.average(x[c], weights=x[w] ) for c in cols], cols)
    except ZeroDivisionError:
        series = pd.Series(0, cols) # set all values to 0 for those months with no prediction.
        return series


# Only used for testing:
def weighted_avg(df, values, weights):
    if df[weights].sum() == 0:
        raise ZeroDivisionError
    return sum(df[values] * df[weights]) / df[weights].sum()

def weighted_means_by_column2(x, cols, w):
    """ This takes a DataFrame and averages each data column (cols)
        while weighting observations by column w.
    """
    try:
        return pd.Series([weighted_avg(x, c, weights=w) for c in cols], cols)
    except ZeroDivisionError:
        series = pd.Series(0, cols) # set all values to 0 for those months with no prediction.
        return series
# ---


def export_dfi(df: pd.DataFrame, path: str) -> None:
    """dfi package tries to export the dataframe with Google Chrome first. On Linux
    this can fail, thus try exporting it with table_conversion=matplotlib."""
    try:
        dfi.export(df, path)
        return
    except OSError:
        print("Exporting performance stats via chrome failed. Trying with "
            "table conversion='matplotlib'...")
    try:
        dfi.export(df, path, table_conversion="matplotlib")
        return
    except OSError as err:
        raise OSError("Try different dataframe .png exporter.") from err


def export_latex(df: pd.DataFrame, path: Path) -> None:
    """Export the dataframe and the dataframe transposed in a .txt file 
    in LaTeX format to the path.
    """
    with path.open("w") as text_file:
        text_file.write(df.style.to_latex())
        text_file.write("\n")
        text_file.write("% Same table transposed:\n")
        text_file.write("\n")
        text_file.write(df.T.style.to_latex())


def get_class_ignore_dates(agg_dict, classes: list, longclass: int) -> dict:
    """For each class get months where the return of the aggreagted portfolio
    is zero. This means that either there was no predicton for that class in that
    month at all or the investment was not made for other reasons (too small of a 
    diversified portfolio, i.e. to few predictions (below an arbitrary cutoff point)).
    
        Returns:
            class_ignore (dict): DatetimeIndeces for each class in a dictionary,
                                 which we were not invested in, in a certain month.
    """
    class_ignore = {}
    for c in classes:
        opt_ret = agg_dict[f"class{c}"]["option_ret"]
        months_no_inv = opt_ret[opt_ret == 0].index
        print_info(classes, c, len(months_no_inv), months_no_inv, longclass) #print info to console.
        class_ignore[f"class{c}"] = months_no_inv
    return class_ignore

def print_info(classes, c, nr_months_no_inv, months_no_inv, longclass):
    if c == classes[0]: #short class, save month indeces to exlude.
        if not nr_months_no_inv:
            print(f"Short Class {c} was invested in every month.")
        else:
            print(f"Short Class {c}, was not invested in the following {nr_months_no_inv} months:", 
                months_no_inv.strftime("%Y-%m-%d").tolist())
    elif c == longclass: #long class, save month indeces to exclude.
        if not nr_months_no_inv:
            print(f"Long Class {c} was invested in every month.")
        else:
            print(f"Long Class {c} was not invested in the following {nr_months_no_inv} months:", 
                months_no_inv.strftime("%Y-%m-%d").tolist())
    else: #remaining classes, just print info.
        if not nr_months_no_inv:
            print(f"Class {c} was invested in every month.")
        else:
            print(f"Class {c}, was not invested in the following {nr_months_no_inv} months:", 
                months_no_inv.strftime("%Y-%m-%d").tolist())

def filter_idx(df: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Filters indeces of 'df' to align with index of 'df_target'.
    
    Returns: Filtered DataFrame.
    """
    return df.loc[df.index[np.isin(df.index, df_target.index)]]


def save_performance_statistics(pf_returns: pd.DataFrame,
                                path_data: Path,
                                path_results_perf: Path,
                                model_name: str, #for txt, png file names.
                                ) -> None:
    """Save performance statistics for the class portfolio returns in pf_returns
    to the 'performance' subfolder in exp_path/'results' """
    # Plot equity line and save to .png in 'path_results_perf'.
    print("Plotting equity lines...")
    pf_returns_cumprod = (1. + pf_returns).cumprod()
    pf_returns_cumprod.plot(figsize=(15, 10), alpha=1.0, linewidth=2.5, ylabel="Portfolio Value",
                    title="Cumulative Return of Classification Portfolios")
    plt.tight_layout() # remove whitespace around plot
    plt.savefig(path_results_perf/f"plot_{model_name}.png")
    print("Done.")

    # Collect performance statistics for EXCESS RETURNS (but cancels for long short portfolio).
    print("Calculating portfolio performance statistics for portfolio (excess) returns...")
    perfstats = []
    periods = 12 # we have monthly (excess) returns (as rf assumed 0.)
    # Convert decimals to percentages of a pandas series.
    def to_pct_string(x):
        """Transforms decimal to % string"""
        return f'{x:.2%}'
    # Cumulative Returns.
    cumr = (pf_returns+1).prod() - 1
    cumr_str = cumr.apply(to_pct_string).rename("Cum. Return") #the to_pct funct. returns a string.
    # Monthly arithmetic mean.
    mean = pf_returns.mean(axis=0)
    mean_str = mean.apply(to_pct_string).rename("Mean (monthly)")
    # Test significance of Long-Short (arithmetic) monthly mean being different from zero.
    # Also with HAC robust errors.
    mean_signif_series = get_save_mean_significance(pf_returns, path_results_perf)
    # Annualized arithmetic mean.
    mean_ann = mean * periods
    mean_ann_str = mean_ann.apply(to_pct_string).rename("Mean (ann.)")
    # Cagr from qs.stats.cagr is wrong (subtracts 1 from lenght of total months, bc it assumes its days)
    cagr = (pf_returns+1).prod() ** (periods/len(pf_returns)) - 1
    cagr_str = cagr.apply(to_pct_string).rename("CAGR")
    # Annualized volatility.
    vol_ann = pf_returns.std(ddof=1) * np.sqrt(periods)
    vol_ann_str = vol_ann.apply(to_pct_string).rename("Volatility (ann.)")
    # Annualized Sharpe Ratio.
    sharpe_ann = mean_ann / vol_ann
    sharpe_ann_str = sharpe_ann.apply(lambda x: f'{x: .3f}').rename("Sharpe Ratio (ann.)")
    # Geometric annualized Sharpe Ratio
    sharpe_ann_geom = cagr / vol_ann
    sharpe_ann_geom_str = sharpe_ann_geom.apply(lambda x: f'{x: .3f}').rename("Geom. Sharpe Ratio (ann.)")
    # Calculate alpha and beta w.r.t. FF Excess Market return.
    print("Load the Market Excess return from the 5 Fama French Factors Dataset...")
    # Skip first two rows (text description) and omit yearly data (after row 706).
    mkt_excess_ret_monthly = load_mkt_excess_ret_monthly(path_data)
    # Filter months (rows) to align with pf_returns.
    mkt_excess_ret_monthly = filter_idx(mkt_excess_ret_monthly, pf_returns)
    print("Done.")
    # Perform linear regression (no HAC adjustment) (alpha then annualized).
    # Convert DataFrame to Series, so that X has correct shape.
    mkt_excess_ret_monthly = mkt_excess_ret_monthly.squeeze()
    # Create X as [[1., 1., ..., 1.].T, [mkt_excess_ret].T] (with intercept).
    X = np.array([np.ones_like(mkt_excess_ret_monthly), mkt_excess_ret_monthly]).T
    alphabetas = np.linalg.inv(X.T@X)@X.T@pf_returns #regress pf_returns (y) on [1, mkt_excess_ret] (X)
    alphas = alphabetas.iloc[0, :]
    alphas_ann = alphas * periods #annualized alpha
    betas = alphabetas.iloc[1, :]
    alphas_ann_str = alphas_ann.apply(lambda x: f'{x: .3f}').rename("Alpha (ann.)")
    alphas_str = alphas.apply(lambda x: f'{x: .3f}').rename("Alpha (monthly.)")
    betas_str = betas.apply(lambda x: f'{x: .3f}').rename("Beta")
    # Test significance of Alpha betas and save detailed ols results.
    # Also with HAC robust errors.
    alpha_signif_df, beta_signif_df = get_save_alphabeta_significance(pf_returns,
                                                        X, 
                                                        path_results_perf, 
                                                        alphas, #check alphas from the "manual" calc.
                                                        betas, #check beta from the "manual" calc.
                                                        )
    # Max Drawdown.
    pf_returns_dd = pf_returns.copy()
    # Insert [0, 0, ..., 0] as first prices, to calculate MaxDD correctly.
    pf_returns_dd.loc[pf_returns_dd.index[0] - MonthEnd(1)] = [0] * pf_returns_dd.shape[1]
    pf_returns_dd = pf_returns_dd.sort_index()
    prices = (1 + pf_returns_dd).cumprod()
    maxdd = (prices / prices.expanding().max()).min() - 1 #formula from quantstats
    maxdd_str = maxdd.apply(to_pct_string).rename("Max Drawdown")
    #Max Drawdown Days.
    maxdd_days = []
    for columname in pf_returns_dd.columns:
        # Important to take 'dfsdd' here and not 'dfs'...
        col_summary = qs.stats.drawdown_details(qs.stats.to_drawdown_series(pf_returns_dd))[columname].sort_values(by="max drawdown", ascending=True)
        # Divide by 100 because qs.stats.drawdown_details provides maxdd in percent.
        qs_dd = col_summary["max drawdown"].iloc[0] / 100
        assert (qs_dd - maxdd[columname]) < 0.0001, (f"Max Drawdown of column {columname}: {qs_dd} calculated by quantstats"
        f"is not equal our manually calculated Max. DD {maxdd[columname]}")
        maxdd_days.append(col_summary["days"].iloc[0])
    maxdd_days = pd.Series(maxdd_days, index=pf_returns_dd.columns, name="Max DD days")
    maxdd_days_str = maxdd_days.apply(lambda x: f'{x: .0f}')
    # Calmar Ratio.
    calmar = cagr / maxdd.abs()
    calmar_str = calmar.apply(lambda x: f'{x: .3f}').rename("Calmar Ratio")
    # Skewness/ Kurtosis.
    skew = pf_returns.skew()
    skew_str = skew.apply(lambda x: f'{x: .3f}').rename("Skewness")
    kurt = pf_returns.kurtosis()
    kurt_str = kurt.apply(lambda x: f'{x: .3f}').rename("Kurtosis")
    # Collect perfstats "strings" (better for saving/ formatting?).
    perfstats += [cumr_str, cagr_str, mean_str, mean_signif_series, mean_ann_str, vol_ann_str, sharpe_ann_str, 
                sharpe_ann_geom_str, maxdd_str, maxdd_days_str, calmar_str, skew_str, 
                kurt_str, alphas_ann_str, alphas_str, alpha_signif_df, betas_str, beta_signif_df
                ]
    print("Save performance statistics to .csv, .png and .txt...")
    # .csv file.
    perfstats = pd.concat(perfstats, axis=1)
    perfstats.columns.name = "Returns are Excess Returns" # Will fill upper left cell.
    perfstats.to_csv(path_results_perf/f"perfstats_{model_name}.csv")
    # dfi only accepts strings as paths.
    export_dfi(perfstats, str(path_results_perf/f"perfstats_{model_name}.png"))
    # Export perfstats dataframe to LaTeX code.
    export_latex(perfstats, path_results_perf/f"perf_latex_{model_name}.txt")