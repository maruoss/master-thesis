from pathlib import Path
import statsmodels.api as sm
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pandas.api.types import is_numeric_dtype

from datamodule import DataModule
from model.neuralnetwork import FFN
from portfolio.helper_perf import check_eoy, get_and_check_min_max_pred, get_class_ignore_dates, various_tests, weighted_means_by_column
from utils.preprocess import YearMonthEndIndeces, binary_categorize, multi_categorize


def get_yearidx_bestmodelpaths(exp_path: Path) -> list:
    """Search exp_path for 'best_ckpt????' files and save their paths into a list.
    Then sort the path and enumerate it to get a list of tuples (idx, best_ckpt_path).

        Returns:
            idx_bestmodelpaths (list): Enumerated best checkpoint paths in the
                                        experiment path.
    """
    best_ckpt_paths = []
    for directory in exp_path.iterdir():
        if directory.is_dir() and directory.name != "predictions" and directory.name != "portfolios":
            # See https://docs.python.org/3/library/fnmatch.html#module-fnmatch
            # for filename pattern matching below.
            for file in directory.glob("best_ckpt????"):
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
                    args_exp: pd.Series
                    ) -> pd.Series:
    """Aggregate orig_feature_df into class portfolios depending on the predictions made in preds_concat_df.
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

    print("Create Long Short Portfolio while ignoring months where one side "
        "is not allocated...")
    # Long-Short PF (highest class (long) - lowest class (short))
    short_class = classes[0] #should be 0
    assert short_class == 0, "Class of short portfolio not 0. Check why."
    long_class = classes[-1] #should be 2 for binary, 3 for 'multi3', etc.
    print(f"Subtract Short portfolio (class {short_class}) from Long portfolio "
            f"(class {long_class})...")
    # Subtract short from long portfolio.
    long_df = agg_dict[f"class{long_class}"].copy() #deep copy to not change original agg_dict
    short_df = agg_dict[f"class{short_class}"].copy() #deep copy to not change original agg_dict
    months_no_inv = class_ignore[f"class{long_class}"].union(class_ignore[f"class{short_class}"]) #union of months to set to 0.
    long_df.loc[months_no_inv, :] = 0
    short_df.loc[months_no_inv, :] = 0
    long_short_df = long_df - short_df #months that are 0 in both dfs stay 0 everywhere.
    assert ((long_short_df.drop(months_no_inv)["pred"] == (long_class - short_class)).all() and #'pred' should be long_class - short_class
            (long_short_df.drop(months_no_inv)["if_long_short"] == 2).all()) #'if_long_short' should be 2 (1 - (-1) = 2)
    print("Done.")
    return long_short_df["option_ret"]


def pred_on_data(
        model_name: str, 
        yearidx: int, 
        bestmodelpath: Path, 
        permuted_feature_target_df: pd.DataFrame,
        args_exp: pd.Series,
        ):
    if model_name == "nn":
        model = FFN.load_from_checkpoint(bestmodelpath)
        dm = DataModule(
            path=None, #Set to None.
            year_idx=yearidx, #Important.
            dataset=None, #Set to None.
            batch_size=100000000, #Predict in one step.
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
        preds = trainer.predict(model=model, datamodule=dm) #returns list of batch predictions.
        preds = torch.cat(preds) #preds is a list already of [batch_size, num_classes]. 
        preds_argmax = preds.argmax(dim=1).numpy()
        preds_argmax_df = pd.DataFrame(preds_argmax, columns=["pred"])

        # y_true (of the whole data available, not only test dates).
        y_new = dm.y.numpy()
        return preds_argmax_df, y_new


def mean_str(col):
    """Takes mean of numeric columns in a dataframe, but handles string 
    columns by returning the count of signif. stars '*'. """
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.str.count("\*").mean()


def mean_str_add_stars(df) -> dict:
    """Take mean of columns of df. '*' will be counted
    int string columns and the mean and the corresponding number of
    '*' will be added in a new column.
    """
    series_mean = df.apply(mean_str)
    num_stars = int(series_mean["Signif."])
    series_mean["SignifStars"] = num_stars * "*"
    return series_mean.to_dict()