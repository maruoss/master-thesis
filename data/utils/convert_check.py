# Manually insert SIC codes
from argparse import ArgumentParser
from pathlib import Path
import time

import pandas as pd


def csv_to_parquet(args):
    """Converts csv file to parquet. Saves new file under same filename."""
    # Convert filenames correctly.
    path = Path(args.path)
    if args.filename[-4:] == ".csv":
        filename = args.filename
    else:
        filename = args.filename + ".csv"
    
    # Read in csv file.
    print("Read in csv...")
    s_time = time.time()
    data = pd.read_csv(path/filename)
    e_time = time.time()
    print("Read csv dataset in: ", (e_time-s_time), "seconds.")

    # Convert to parquet.
    print("Convert to parquet...")
    # assert filename[:-4] == ".csv", "filename muss be a .csv file"
    name_nosuffix = filename[:-4] #remove .csv
    parquet_name = name_nosuffix+".parquet"
    data.to_parquet(path/parquet_name)
    print("Done.")

def csv_eq_parquet(args):
    """Check whether loaded pandas dataframes from 'filename'.csv and 
    'filename.parquet' are equal."""
    # Convert path to Pathlib path. If already (default), will leave as is.
    path = Path(args.path)

    # If filename already has suffixes remove them, otherwise leave as is.
    if args.filename[-4:] == ".csv" or args.filename[-4:] == ".parquet":
        filename = args.filename[:-4]
    else:
        filename = args.filename

    # Read in 'filename'.csv.
    filename_csv = filename + ".csv"
    s_time = time.time()
    df_from_csv = pd.read_csv(path/filename_csv)
    e_time = time.time()
    print("Read csv dataset in: ", (e_time-s_time), "seconds.")

    # Read in 'filename'.parquet.
    filename_parq = filename + ".parquet"
    s_time = time.time()
    df_from_parq = pd.read_parquet(path/filename_parq)
    e_time = time.time()
    print("Read parquet dataset in: ", (e_time-s_time), "seconds.")

    # Compare both dataframes.
    if df_from_csv.equals(df_from_parq) and df_from_parq.equals(df_from_csv):
        print("Both datasets are the same.")
    else:
        raise ValueError("Datasets are not the same.")


def small_med_big_eq(path: Path):
    """Read in 'small', 'medium' and 'big' datasets and check whether their dates
    and option return columns equal -> whether we predicted on same row order 
    for all three datasets...
    
    Args:
        path (Path):    requires 'final_df_call_cao_small.parquet', 'final_df_call_
                        cao_med_fillmean.parquet' and 'final_df_call_cao_big_fillmean.parquet' 
                        files in it.
    Returns:
        bool:           True, when dates and returns of all files are equal.
                        False, when dates and returns of files are NOT equal.
    """
    # Read in 'small', 'med' and 'big' datasets"
    print("Read in csv's...")
    s_time = time.time()
    df_small = pd.read_parquet(path/"final_df_call_cao_small.parquet")
    df_small = df_small.loc[:, ["date", "option_ret"]]
    df_med = pd.read_parquet(path/"final_df_call_cao_med_fillmean.parquet")
    df_med = df_med.loc[:, ["date", "option_ret"]]
    df_big = pd.read_parquet(path/"final_df_call_cao_big_fillmean.parquet")
    df_big = df_big.loc[:, ["date", "option_ret"]]
    e_time = time.time()
    print("Read 'small', 'medium' and 'big' datasets "
            "in", (e_time-s_time), "seconds.")
    
    # Compare them, make sure they are all equal. (order should be preserved from
    # original dataset)
    if (df_small.equals(df_med) and df_med.equals(df_big) and df_big.equals(df_small)): 
        return True
    else:
        # Files are not equal.
        return False


if __name__ == "__main__":
    parser = ArgumentParser(description="Helper to convert csv files to the "
            "parquet format, which is much faster to read/ write.")

    path = Path.cwd()/"data"
    # only nargs='?' allows to omit positional keyword argument
    parser.add_argument("function", choices=["csv_to_parquet", "csv_eq_parquet", "check"])
    parser.add_argument("-f", "--filename", type=str) #saved as args.filename
    parser.add_argument("--path", type=str, default=path)

    args = parser.parse_args()

    if args.function == "csv_to_parquet":
        csv_to_parquet(args)
    elif args.function == "csv_eq_parquet":
        csv_eq_parquet(args)
    elif args.function == "check":
        assert small_med_big_eq(path), "Dates and option returns of small, medium and big datasets are NOT equal!"
        print("Success! Dates and option returns of small, medium and big datasets are equal!")
