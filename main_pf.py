from argparse import ArgumentParser
from operator import concat
from pathlib import Path
import pdb
import numpy as np
import pandas as pd

from data.utils.convert_check import small_med_big_eq
from portfolio.helper import check_eoy, collect_preds, concat_and_save_preds, get_and_check_min_max_pred, various_tests, weighted_avg, weighted_means_by_column, weighted_means_by_column2
from utils.preprocess import YearMonthEndIndeces


def run(args):
    # Get experiment folder.
    logs_folder = Path.cwd()/"logs"
    matches = Path(logs_folder).rglob(args.expid) #Get folder in logs_folder that matches expid
    matches_list = list(matches)
    assert len(matches_list) == 1, "there exists more than 1 folder with given expid!"
    exp_dir = matches_list[0]
    # Move all predictions to 'predictions' folder with the expdir folder.
    collect_preds(exp_dir)
    #TODO: combine (ensemble) of multiple experiment predictions together? list of expids?
    # Read all prediction csv in concatenated df.
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

    # Create separate weight column for each class.
    for c in classes:
        condlist = [concat_df["pred"] == c]
        choicelist = [1]
        no_alloc_value = 0
        concat_df[f"weights_{c}"] = np.select(condlist, choicelist, no_alloc_value)

    # Only calculate weighted average for numerical columns.
    col_list = [val for val in concat_df.columns.tolist() if "weight" not in val 
                and "date" not in val]

    # Collect all portfolios in a list.
    agg_dict = {}
    for c in classes:
        agg_df = concat_df.groupby("date").aggregate(weighted_means_by_column, col_list, f"weights_{c}")
        agg_dict[f"class_{c}"] = agg_df

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
    # long_short_monthly = (agg_dict[f"class_{classes[-1]}"] - agg_dict[f"class_{classes[0]}"])
    # cols_to_keep = [col for col in long_short_monthly.columns.tolist() if "weight" not in col]
    # long_short_monthly = long_short_monthly[cols_to_keep]

    # TODO: create LaTeX output for portfolio mean, sd, sharpe ratio as in Bali (2021)?
    pdb.set_trace()


if __name__ == "__main__":
    parser = ArgumentParser(description=
        "Master Thesis Mathias Ruoss - Option Return Classification: "
        "Create portfolios from predictions."
    )
    parser.add_argument("expid", type=str, help="folder name of experiment, "
                        "given by time created")

    args = parser.parse_args()
    
    run(args)
