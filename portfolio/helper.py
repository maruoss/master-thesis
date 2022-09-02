from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import dataframe_image as dfi
from tqdm import tqdm

def collect_preds(exp_dir: Path):
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


def export_dfi(perfstats: pd.DataFrame, path: str) -> None:
    """dfi tries exporting the dataframe with Google Chrome first. On Linux
    this can fail, so then it tries exporting with table_conversion=matplotlib."""
    try:
        dfi.export(perfstats, path)
        return
    except OSError:
        print("Exporting performance stats via chrome failed. Trying with "
            "table conversion='matplotlib'...")
    try:
        dfi.export(perfstats, path, table_conversion="matplotlib")
        return
    except OSError as err:
        raise OSError("Try different dataframe .png exporter.") from err


def get_class_ignore_dates(concat_df: pd.DataFrame, classes: list) -> dict:
    """For each class get months where there was no prediction for it at all.
    
        Returns:
            class_ignore (dict): DatetimeIndeces for each class in a dictionary.
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