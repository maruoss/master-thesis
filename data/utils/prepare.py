from argparse import ArgumentParser, Namespace
from pathlib import Path
import pdb
import time
import pandas as pd
from tqdm import tqdm

from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sic_codes import add_sic_manually


def prepare_dataset(args):
    # Convert string path to pathlib path.
    path = Path(args.path)
    # Check whether required files exist in the path.
    assert (path/"sp500_op_ret.parquet").exists(), (
            "'sp500_opt_ret.parquet' does not exist in data path. "
            "If only .csv file is available, convert first using the helper "
            "function 'csv_to_parquet'."
            )
    assert (path/"mapping_table.csv").exists(), ("'mapping_table.csv' does not exist "
                                                "in data path")
    assert (path/"datashare.parquet").exists(), (
        "'datashare.parquet' does not exist in data path. "
        "If only .csv file is available, convert first using the helper "
        "function 'csv_to_parquet'.")

    # Load S&P 500 option returns.
    print("Read in S&P500 option returns from parquet file...")
    s_time = time.time()
    sp500opt_df = pd.read_parquet(path/"sp500_op_ret.parquet")
    e_time = time.time()
    print("Done! Read 'sp500_op_ret.parquet' file in:", (e_time-s_time), "seconds.")

    sp500opt_df["date"] = pd.to_datetime(sp500opt_df["date"])
    sp500opt_df["exdate"] = pd.to_datetime(sp500opt_df["exdate"])
    # Bring dates to end of month dates (in order to correctly merge with other datasets).
    sp500opt_df["date"] = sp500opt_df["date"] + MonthEnd(0)
    # Remove 138 NaN option returns.
    sp500opt_df = sp500opt_df.drop(sp500opt_df[sp500opt_df["option_ret"].isnull()].index)
    sp500opt_df = sp500opt_df.reset_index(drop=True)

    # Load mapping table data (options <-> underlying stocks). Note: The last
    # 7 lines (line 28278-28284) in the .csv. were added manually to fix minor
    # date gaps.
    print("Read in mapping table that maps S&P 500 option returns with underlying stocks...")
    s_time = time.time()
    map_table_df = pd.read_csv(path/"mapping_table.csv") # file is small enough to use .csv
    e_time = time.time()
    print("Done! Read 'mapping_table.csv' in:", (e_time-s_time), "seconds.")

    # Merge Option Returns and mapping table on "secid".
    print("Merge S&P 500 option returns with underlying stock 'secid'...")
    opt_df = sp500opt_df.merge(map_table_df, on="secid")
    print("Done!")

    # Correct format important here, otherwise will see month as day and vice versa.
    print("Filter out dates that do not lie within start and end dates...")
    opt_df["sdate"] = pd.to_datetime(opt_df["sdate"], format = "%d/%m/%Y %H:%M")
    opt_df["edate"] = pd.to_datetime(opt_df["edate"], format = "%d/%m/%Y %H:%M")

    # Filter opt_df to only contain dates that are within 'sdate' and 'edate'.
    # 'date' is used instead of 'exdate', more rows are preserved -> seems more logical.
    opt_df = opt_df[(opt_df["date"] >= opt_df["sdate"]) & 
                    ((opt_df["date"] <= opt_df["edate"]) | 
                    (opt_df["edate"] >= "2020-12-31"))]
    print(f"Done! The merged df after filtering has {len(sp500opt_df)-len(opt_df)} "
        "rows less than the original S&P 500 option return dataset.")
    # 4 observations are dropped because their date is after the 'edate' of the 
    # mapping list...

    # Load Gu2020 stock data.
    print("Load Gu2020 data...")
    s_time = time.time()
    gu2020_df = pd.read_parquet(path/"datashare.parquet")
    e_time = time.time()
    print("Done! Read in:", (e_time-s_time), "seconds.")

    # Lag features by -1 to align with option features. See readme file of Gu2020
    # datashare directory. They aligned their data by pushing features forward.
    # Now, we have to push them back again, since in our data the return was lagged.
    print("Clean Gu2020 data...")
    gu2020_df["DATE"] = pd.to_datetime(gu2020_df["DATE"].astype(str), 
                                        format='%Y%m%d') - MonthEnd(1)

    # Adds missing sic codes manually, after verifying them with given sources.
    gu2020_df = add_sic_manually(gu2020_df)

    # Our data starts at 1996-01-31.
    gu2020_df = gu2020_df[gu2020_df["DATE"] > "1995-12-31"]
    gu2020_df = gu2020_df.reset_index()
    gu2020_df = gu2020_df.rename(columns={"DATE": "date"})

    if args.fill_na:
        # Fill missing values.
        print(f"Fill missing values... There are {gu2020_df.isnull().to_numpy().sum()} "
                f"missing values in the Gu2020 dataset. Fill them with {args.fill_fn} of stocks of "
                "respective month...")
        # Unique set of dates to loop over and take mean/median of all stocks in that month.
        date_unique = list(gu2020_df["date"].unique())
        # Loop over each month and fill NA with stock mean/median of that month.
        if args.fill_fn == "mean":
            for i in tqdm(date_unique):
                gu2020_df[gu2020_df["date"] == i] = (gu2020_df[gu2020_df["date"] == i]
                                                    .fillna(gu2020_df[gu2020_df["date"] == i]
                                                    .mean(axis=0, numeric_only=True)))
        elif args.fill_fn == "median":
            for i in tqdm(date_unique):
                gu2020_df[gu2020_df["date"] == i] = (gu2020_df[gu2020_df["date"] == i]
                                                    .fillna(gu2020_df[gu2020_df["date"] == i]
                                                    .median(axis=0, numeric_only=True)))
        else:
            raise ValueError("Select mean or median as fill fn.")

        # Check if there are NaN values remaining.
        assert opt_df.isnull().to_numpy().sum() == 0, "Not all NaN values of opt_df have been dropped."
        assert gu2020_df.isnull().to_numpy().sum() == 0, "Not all NaN values of gu2020_df have been filled."
        print("Done!")

    print(f"There are now {gu2020_df.isnull().to_numpy().sum()} missing values.")
    # Merge option data with underlying Gu2020 stock data.
    # "inner" removes about 2000 rows. "left" would have these as NaNs.
    print("Merge option data with underlying Gu2020 stock data...")
    final_df = opt_df.merge(gu2020_df, on=["date", "permno"], how="inner")
    print(f"Done! {len(opt_df)-len(final_df)} rows have been removed by the "
          "inner join of the option and stock data on 'date' and 'permno'.")
    # Put option returns at the end.
    opt_ret = final_df.pop("option_ret")
    final_df["option_ret"] = opt_ret

    # Transform Categorical variables to OneHot-Numeric array
    print("Transform categorical variables cp_flag and sic2 code to OneHot-numeric array "
        "and drop useless columns.")
    onehotencoder = make_column_transformer((OneHotEncoder(), 
                                            ["cp_flag", "sic2"]), remainder="drop")
    transformed = onehotencoder.fit_transform(final_df)
    transformed_df = pd.DataFrame(transformed.toarray(), 
                                    columns=onehotencoder.transformers_[0][1]
                                    .get_feature_names_out(["cp_flag", "sic2"]))
    # Drop old columns cp_flag, sic2.
    final_df = final_df.drop(["cp_flag", "sic2"], axis=1)
    # Concatenate both dataframes (memory intensive?)
    final_df = pd.concat([final_df, transformed_df], axis=1)
    # Sort values by date! (secondly by secid -> but better to shuffle within date later?)
    final_df = final_df.sort_values(["date", "secid"]).reset_index(drop=True)
    # Remove useless columns for data analysis.
    final_df = final_df.drop(["secid", "optionid", "exdate", "sdate", "edate", "permno", "index"], axis=1)
    print("Done!")

    # Save final_df depending on arguments.
    print("Save final dataframe...")
    save_df(path, final_df, args)
    print("Done!")
    print("---")
    print("All done.")


def save_df(path: Path, final_df: pd.DataFrame, args: Namespace):
    # Create and save small, medium and big datasets.
    # Small dataset takes the option_return data + the newly created one_hot columns 
    # + option return, gives in total 22 columns (incl. target, option_return)
    prefix = f"final_df{args.tag}"
    if args.size == "small":
        filename = prefix + "_small.parquet"
        # Only keep first 19 columns + 'cp_flag_C', 'cp_flag_P' and the target 'option_ret'
        small_df_columns = (final_df.iloc[:, :19].columns
                        .append(final_df.loc[:, ["cp_flag_C", "cp_flag_P", "option_ret"]].columns))
        small_final_df = final_df.loc[:, small_df_columns]
        small_final_df.to_parquet(path/filename) # 22 columns.
    
    elif args.size == "medium":
        # Medium dataset takes all option + stock data but no sic onehot columns.
        if args.fill_na:  # NaNs filled.
            filename = prefix + f"_med_fill{args.fill_fn}.parquet"
        else: # NaNs not filled.
            filename = prefix + "_med_nofill.parquet"
        # Remove all sic codes from big dataset to get medium.
        medium_final_df = final_df.loc[:, ~final_df.columns.str.startswith('sic2')]
        medium_final_df.to_parquet(path/filename) # 116 columns.

    elif args.size == "big":
        # Big dataset leaves data as it is.
        if args.fill_na: # NaNs filled.
            filename= prefix + f"_big_fill{args.fill_fn}.parquet"
        else: # NaNs not filled
            filename = prefix + "_big_nofill.parquet"
        final_df.to_parquet(path/filename) # 179 columns.

    print(f"The final dataframe with specifications:\nsize: {args.size},\nfill_na: {args.fill_na}, "
          f"\nfill_fn: {args.fill_fn}\nhas been saved to {path/filename} .")


if __name__ == "__main__":
    parser = ArgumentParser(description="Loads 'sp500_opt_ret.parquet', 'mapping_table.csv'" 
    "and 'datashare.parquet' files. Merges them according to the "
    "specified final target datasize 'size' and fills NaN values of the Gu "
    "dataset according to the specified function.")
     
    # Data path where 'sp500_opt_ret.parquet', 'mapping_table.csv' and 
    # 'datashare.parquet' should reside.
    path = Path.cwd()/"data"

    parser.add_argument("size", type=str, choices=["small", "medium", "big"], 
                        help="Size of the desired final dataset.")
    parser.add_argument("--path", type=str, 
                    default=path,
                    help="Path where all the three required data files should lie.")
    pdb.set_trace()
    parser.add_argument("--fill_fn", type=str, default="mean", choices=["mean", "median"],
                        help="whether mean or median should be used for filling NaN values")
    parser.add_argument("--fill_na", action="store_true", help="fill na values "
                        "of the Gu2020 dataset.")
    parser.add_argument('--no-nafill', dest='fill_na', action='store_false',
                        help="dont fill na values of Gu2020 dataset.")
    parser.set_defaults(fill_na=True)
    parser.add_argument("--tag", type=str, default="", help="tag to add to final df parquet files")

    args = parser.parse_args()

    prepare_dataset(args)