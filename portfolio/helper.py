from pathlib import Path
import shutil
import numpy as np

import pandas as pd


def collect_preds(exp_dir: Path):
    """Moves all predictions????.csv to a 'predictions' folder within the
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
                        shutil.copy(file, preds_dir)
                    except shutil.SameFileError:
                        print("Source and Destination are the same file...")
                else:
                    print(f"File {file.name} already exists in '{preds_dir.name}' folder.")


def concat_and_save_preds(exp_dir: Path):
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


def check_month_years(dic, dates):
    """Checks whether all end of month indeces in the dictionary 'dic'
    are in the correct year. Also, checks whether all indeces are in
    consecutive order. 31.12.2019[31.01.2020,......,31.12.2020, 31.01.2021]
    
    The last month of the eom indeces overlaps with the first entry in
    the next year.

    ---
    Example:
        If a year has 12 months in the data, the end of month indeces should 
        have length 13. The first index is the first "row" of the year, 
        the last index is the first row of the next year.
    """
    for year in dic.keys():
        len_dic = len(dic[year])
        for idx, eom_idx in enumerate(dic[year]):
            # Special case: last eom_idx is first eom_idx of next year.
            if idx == len_dic - 1: #idx uses zero indexing.
                if int(year) != dates[eom_idx-1].year or (idx)!= dates[eom_idx-1].month:
                    return False
            elif int(year) != dates[eom_idx].year or (idx+1) != dates[eom_idx].month:
                return False
    return True

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
    """Requires concatenated df with option returns and its predictions. labelfn_exp
    is the label_fn of the experiment and should be a string 'binary' or 
    'multi{num_classes}'."""
    # Min pred value theoretically.
    min_theor = 0
    if labelfn_exp=="binary":
        max_theor = 1 
    else: #multi3, multi5, multi10 -> take (3, 5, 10) - 1
        max_theor = int(labelfn_exp[5:]) - 1 # 3 classes -> 0, 1, 2
    # Max pred value realized per month.
    max_real_series = concat_df.groupby("date")["pred"].max()
    max_real = max_real_series.max()
    print("Max prediction realized is:", max_real)
    assert (max_real == max_real_series).all() and max_theor == max_real, (
        f"Not all max class predictions are equal in each month or "
        f"the maximum class {max_theor} was not predicted in at least one month.")
    # Min pred value realized per month.
    min_real_series = concat_df.groupby("date")["pred"].min()
    min_real = min_real_series.min()
    print("Min prediction realized is:", min_real)
    assert (min_real == min_real_series).all() and min_theor == min_real, (
                f"Not all min class predictions are equal in each month "
                f"the minimum class {min_theor} was not predicted in at least one month.")
    return max_real, min_real


def various_tests(concat_df: pd.DataFrame, col_list: list, classes: list, agg_dict: dict):
    """Perform various tests to check our results."""

    # Test: compare agg_dict with agg_dict calculated via 'weighted_avg' function 
    # and not via 'np.average'.
    agg_dict2 = {}
    for c in classes:
        agg_df = concat_df.groupby("date").aggregate(weighted_means_by_column2, col_list, f"weights_{c}")
        agg_dict2[f"class{c}"] = agg_df
    for key in agg_dict.keys():
        pd.testing.assert_frame_equal(agg_dict[key], agg_dict2[key])

    # Test: check whether first and last month aggregation yield same as
    # first and last entries of agg_dict.
    first_month = concat_df.loc[concat_df["date"] == concat_df["date"].iloc[0]]
    last_month = concat_df.loc[concat_df["date"] == concat_df["date"].iloc[-1]]
    for k in col_list:
        for c in classes:
            assert np.average(first_month[k], weights=first_month[f"weights_{c}"]) == agg_dict[f"class{c}"].iloc[0][k]
            assert np.average(last_month[k], weights=last_month[f"weights_{c}"]) == agg_dict[f"class{c}"].iloc[-1][k]
            assert weighted_avg(first_month, k, f"weights_{c}") == agg_dict2[f"class{c}"].iloc[0][k]
            assert weighted_avg(last_month, k, f"weights_{c}") == agg_dict2[f"class{c}"].iloc[-1][k]

    # Test: if "pred" column in aggregated df's corresponds to class in each row (month).
    for c in classes:
        assert (agg_dict[f"class{c}"]["pred"] == c).all(), "Aggregated 'pred' is not equal to the class in at least one month."
    # Test if short and low portfolios are aggregated correctly.
    assert ((agg_dict[f"class{classes[0]}"]["if_long_short"] == -1).all() and
            (agg_dict[f"class{classes[-1]}"]["if_long_short"] == 1).all()), ("Long "
            "or short portfolio aggregation does not yield 1 or -1 in 'if_long_short' column.")


# Weighted average functions used to aggreagte portfolios. We use np.average.
def weighted_means_by_column(x, cols, w):
    """ This takes a DataFrame and averages each data column (cols)
        while weighting observations by column w.
    """
    return pd.Series([np.average(x[c], weights=x[w] ) for c in cols], cols)


# Only used for testing:
def weighted_avg(df, values, weights):
    return sum(df[values] * df[weights]) / df[weights].sum()

def weighted_means_by_column2(x, cols, w):
    """ This takes a DataFrame and averages each data column (cols)
        while weighting observations by column w.
    """
    return pd.Series([weighted_avg(x, c, weights=w) for c in cols], cols)
# ---