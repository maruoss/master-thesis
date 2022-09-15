from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import dataframe_image as dfi
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
import quantstats as qs
import statsmodels.api as sm

from portfolio.load_files import load_mkt_excess_ret_monthly
from portfolio.utils import add_signif_stars_df


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
                    print(f"File {file.name} already exists in '{preds_dir.name}' folder.")


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
    print(f"Number of months where Short class {min_real} is not predicted:", 
            months_no_min, "out of", f"{len(min_real_series)}.")
    # Max pred value realized per month.
    max_real_series = concat_df.groupby("date")["pred"].max()
    max_real = max_real_series.max()
    # max_real_series[max_real_series != max_real].index.strftime("%Y-%m-%d").to_list()
    assert max_theor == max_real, (
        "Not a single month has the prediction of the theoretical maximum class.")
    months_no_max = max_real_series[max_real_series != max_real].count()
    print(f"Number of months where Long class {max_real} is not predicted:", 
            months_no_max, "out of", f"{len(max_real_series)}.")
    return max_real, min_real, classes


def various_tests(agg_dict: dict, concat_df: pd.DataFrame, col_list: list, classes: list, class_ignore: dict):
    """Perform various sanity checks on our monthly aggregated results in agg_dict."""
    # Test1: Compare agg_dict with agg_dict2, calculated via 'weighted_avg' function 
    # and not via 'np.average'. They should yield the same (up to small precision).
    agg_dict2 = {}
    for c in tqdm(classes):
        agg_df = concat_df.groupby("date").aggregate(weighted_means_by_column2, col_list, f"weights_{c}")
        agg_dict2[f"class{c}"] = agg_df
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
            (agg_dict_copy[f"class{classes[-1]}"]["if_long_short"] == 1).all()), ("Long "
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
                elif c==classes[-1]:
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


def export_latex(df: pd.DataFrame, path_results_perf: Path) -> None:
    """Export the dataframe and the dataframe transposed in a .txt file 
    in LaTeX format to the 'path_results_perf' path.
    """
    with (path_results_perf/"perf_latex.txt").open("w") as text_file:
        text_file.write(df.style.to_latex())
        text_file.write("\n")
        text_file.write("% Same table transposed:\n")
        text_file.write("\n")
        text_file.write(df.T.style.to_latex())
    print("Done.")


def get_class_ignore_dates(concat_df: pd.DataFrame, classes: list) -> dict:
    """For each class get months where there was no prediction for it at all
    in the dataframe.
    
        Returns:
            class_ignore (dict): DatetimeIndeces for each class in a dictionary,
                                 which did not have a prediction in a certain month.
    """
    class_ignore = {}
    for c in classes:
        sum_onehot = concat_df.groupby("date")[f"weights_{c}"].sum()
        nr_months_noclass = sum_onehot[sum_onehot==0].count()
        months_noclass = sum_onehot[sum_onehot==0].index #Datetimeindex of months.
        if c == classes[0]: #short class, save month indeces to exlude.
            if not nr_months_noclass:
                print(f"Short Class {c} was predicted in every month.")
            else:
                print(f"Short Class {c}, was not predicted in the following {nr_months_noclass} months:", 
                months_noclass.strftime("%Y-%m-%d").tolist())
        elif c == classes[-1]: #long class, save month indeces to exclude.
            if not nr_months_noclass:
                print(f"Short Class {c} was predicted in every month.")
            else:
                print(f"Long Class {c} was not predicted in the following {nr_months_noclass} months:", 
                months_noclass.strftime("%Y-%m-%d").tolist())
        else: #remaining classes, just print info.
            if not nr_months_noclass:
                print(f"Class {c} was predicted in every month.")
            else:
                print(f"Class {c}, was not predicted in the following {nr_months_noclass} months:", 
                months_noclass.strftime("%Y-%m-%d").tolist())
        class_ignore[f"class{c}"] = months_noclass
    return class_ignore


def filter_idx(df: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Filters indeces of 'df' to align with index of 'df_target'.
    
    Returns: Filtered DataFrame.
    """
    return df.loc[df.index[np.isin(df.index, df_target.index)]]


def save_performance_statistics(pf_returns: pd.DataFrame, 
                                exp_path: Path, 
                                path_data: Path,
                                path_results_perf: Path
                                ) -> None:
    """Save performance statistics for the class portfolio returns in pf_returns
    to the 'performance' subfolder in exp_path/'results' """
    # Plot equity line and save to .png.
    print("Plotting equity lines...")
    pf_returns_cumprod = (1. + pf_returns).cumprod()
    pf_returns_cumprod.plot(figsize=(15, 10), alpha=1.0, linewidth=2.5, ylabel="Portfolio Value",
                    title="Cumulative Return of Classification Portfolios")
    plt.tight_layout() # remove whitespace around plot
    # # Make 'results' subfolder.
    # path_results = exp_path/"results"
    # path_results.mkdir(exist_ok=True, parents=False)
    # # Make 'performance' subfolder in 'results' folder.
    # path_results_perf = path_results/"performance"
    # path_results_perf.mkdir(exist_ok=True, parents=False)
    plt.savefig(path_results_perf/"plot.png")
    print("Done.")

    # Collect performance statistics for EXCESS RETURNS (but cancels for long short portfolio).
    print("Calculating portfolio performance statistics for portfolio (excess) returns...")
    perfstats = []
    periods = 12 # we have monthly (excess) returns (as rf assumed 0.)

    # Convert decimals to percentages of a pandas series.
    def to_pct(x):
        """Transforms decimal to % string"""
        return f'{x:.2%}'

    # Cumulative Returns.
    cumr = (pf_returns+1).prod() - 1
    cumr_str = cumr.apply(to_pct).rename("Cum. Return")

    # Annualized arithmetic mean.
    mean_ann = pf_returns.mean() * periods
    mean_ann_str = mean_ann.apply(to_pct).rename("Mean (ann.)")

    # Cagr from qs.stats.cagr is wrong (subtracts 1 from lenght of total months, bc it assumes its days)
    cagr = (pf_returns+1).prod() ** (periods/len(pf_returns)) - 1
    cagr_str = cagr.apply(to_pct).rename("CAGR")

    # Annualized volatility.
    vol_ann = pf_returns.std(ddof=1) * np.sqrt(periods)
    vol_ann_str = vol_ann.apply(to_pct).rename("Volatility (ann.)")

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

    # Perform linear regression (no HAC adjustment). Should yield same as OLS 
    # Standard later (but here annualized).
    # Convert DataFrame to Series, so that X has correct shape.
    mkt_excess_ret_monthly = mkt_excess_ret_monthly.squeeze()
    X = np.array([np.ones_like(mkt_excess_ret_monthly), mkt_excess_ret_monthly]).T #bring to shape [1, mkt_excess_ret]
    alphabetas = np.linalg.inv(X.T@X)@X.T@pf_returns #regress pf_returns (y) on [1, mkt_excess_ret] (X)
    alphas = alphabetas.iloc[0, :] * periods #annualized alpha
    betas = alphabetas.iloc[1, :]
    alphas_str = alphas.apply(lambda x: f'{x: .3f}').rename("Alpha (ann.)")
    betas_str = betas.apply(lambda x: f'{x: .3f}').rename("Beta")

    # Max Drawdown.
    pf_returns_dd = pf_returns.copy()
    # Insert [0, 0, ..., 0] as first prices, to calculate MaxDD correctly.
    pf_returns_dd.loc[pf_returns_dd.index[0] - MonthEnd(1)] = [0] * pf_returns_dd.shape[1]
    pf_returns_dd = pf_returns_dd.sort_index()
    prices = (1 + pf_returns_dd).cumprod()
    maxdd = (prices / prices.expanding().max()).min() - 1 #formula from quantstats
    maxdd_str = maxdd.apply(to_pct).rename("Max Drawdown")

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
    perfstats += [cumr_str, cagr_str, mean_ann_str, vol_ann_str, sharpe_ann_str, 
                sharpe_ann_geom_str, maxdd_str, maxdd_days_str, calmar_str, skew_str, 
                kurt_str, alphas_str, betas_str
                ]

    print("Save performance statistics to .csv, .png and .txt...")
    # Save results to 'results' subfolder.
    perfstats = pd.concat(perfstats, axis=1)
    perfstats.to_csv(path_results_perf/"perfstats.csv")
    # dfi only accepts strings as paths:
    export_dfi(perfstats, str(path_results_perf/"perfstats.png"))

    # Export latex code for table.
    with (path_results_perf/"perf_latex.txt").open("w") as text_file:
        text_file.write(perfstats.style.to_latex())
        text_file.write("\n")
        text_file.write("% Same table transposed:\n")
        text_file.write("\n")
        text_file.write(perfstats.T.style.to_latex())
    print("Done.")


def regress_factors(regression_map: dict, factors_avail: pd.DataFrame, y: pd.Series, path_results: Path) -> None:
    """Performs each relevant regression from the 'regression_map' dictionary and 
    saves results as .txt (latex) and .csv files. in an accordingly named folder
    in 'path_results'. 

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
        fileprefix = "Standard"
        save_ols_results(ols_result, parent_path, fileprefix)

        # 1b. Regression. With HAC Standard Errors. Lags of Greene (2012): L = T**(1/4).
        max_lags = int(len(X)**(1/4))
        ols_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}, use_t=True)
        fileprefix = f"HAC_{max_lags}"
        save_ols_results(ols_result, parent_path, fileprefix)

        # 1c. Regression. With HAC Standard Errors. Lag after Bali (2021) = 12.
        ols_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": 12}, use_t=True)
        fileprefix = "HAC_12"
        save_ols_results(ols_result, parent_path, fileprefix)


def save_ols_results(ols_result, parent_path: Path, fileprefix: str) -> None:
    """Saves results from statsmodels.OLS to .txt and .csv files in a separate 
    folder in the 'parent_path' path.

        Args:
            ols_result:             Object from ols.fit().summary()
            ols_result2:            Object from ols.fit().summary2()
            parent_path (Path):     Path to the results folder.
            fileprefix (str):       Prefix name of the file.
    
    """
    # 3 LaTeX files.
    # with (parent_path/f"{fileprefix}_result_latex1.txt").open("w") as text_file:
    #     text_file.write("OLS Summary (For Loop):\n\n")
    #     for table in ols_result.summary(alpha=0.05).tables:
    #         text_file.write(table.as_latex_tabular())
    # NOTE: summary() doesnt work with adding significance stars + layout is not as nice.
    with (parent_path/f"{fileprefix}_latex.txt").open("w") as text_file:        
        text_file.write("OLS Summary2 as LaTeX:\n\n")
        summary2 = ols_result.summary2(alpha=0.05)
        summary2.tables[1] = add_signif_stars_df(summary2.tables[1])
        summary2.tables[0] = replace_scale_with_cov_type(summary2.tables[0], 
                                                        fileprefix)
        text_file.write(summary2.as_latex())
        text_file.write("\n\n\nOLS Summary as LaTeX:\n\n")
        text_file.write(ols_result.summary(alpha=0.05).as_latex())
    # .txt file.
    with (parent_path/f"{fileprefix}.txt").open("w") as text_file:
        text_file.write(summary2.as_text())
    # 1 .csv file.
    with (parent_path/f"{fileprefix}_summary1.csv").open("w") as text_file:
        text_file.write(ols_result.summary(alpha=0.05).as_csv())
    # 2. .csv file
    with (parent_path/f"{fileprefix}_summary2.csv").open("w") as text_file:
        for idx, df in enumerate(summary2.tables):
            # Only print rows and cols for table1, where they are annotated.
            if idx==1: 
                df.to_csv(text_file, line_terminator="\n")
            else:
                df.to_csv(text_file, line_terminator="\n", index=False, header=False)
            # text_file.write("\n")



def replace_scale_with_cov_type(ols_summary2_table0: pd.DataFrame, 
                                cov_type: str
                                ) -> pd.DataFrame:
    """Summary2 does not have the cov type by default in its tables. Here we replace
    the 'Scale' statistic at location (6, 2) and (6, 3) with the covariance type."""
    ols_summary2_table0.iloc[6, 2] = "Cov. Type"
    ols_summary2_table0.iloc[6, 3] = cov_type
    return ols_summary2_table0