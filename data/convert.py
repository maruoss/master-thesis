# Manually insert SIC codes
from argparse import ArgumentParser
from pathlib import Path
import time

import pandas as pd


def csv_to_parquet(args):
    """Converts csv file to parqurt. Saves new file under same filename."""
    print("Read in csv...")

    path = Path(args.path)
    filename = args.filename
    
    s_time = time.time()
    data = pd.read_csv(path/filename)
    e_time = time.time()
    print("Read csv dataset in: ", (e_time-s_time), "seconds.")

    print("Convert to parquet...")
    # assert filename[:-4] == ".csv", "filename muss be a .csv file"
    name_nosuffix = filename[:-4] #remove .csv
    parquet_name = name_nosuffix+".parquet"
    data.to_parquet(path/parquet_name)
    print("Done.")


if __name__ == "__main__":

    parser = ArgumentParser(description="Helper to convert csv files to the parquet "
            "format, which is much fast to read/ write.")

    path = r"C:\Users\Mathiass\OneDrive - Universität Zürich UZH\Documents\mt_literature\data"
    parser.add_argument("filename", type=str)
    parser.add_argument("--path", type=str, default=path)

    args = parser.parse_args()

    csv_to_parquet(args)