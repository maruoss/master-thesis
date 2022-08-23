from argparse import ArgumentParser
from pathlib import Path
import pdb
import pandas as pd
import yfinance as yf

from datetime import timedelta
from pandas.tseries.offsets import MonthEnd

def download_sp500TR_prices(args):
    """Downloads monthly sp500 total return index prices from Yahoo Finance."""
    path = Path(args.path)
    tickers = ["^SP500TR"]
    try:
        opt_ret = pd.read_parquet(path/"sp500_op_ret.parquet")
    except FileNotFoundError as err:
        raise FileNotFoundError("Make sure to be in the main project working directory.") from err

    # For monthly returns we need end of month prices -> go back suffic. time to get price of month end -1.
    start = pd.to_datetime(opt_ret["date"]).iloc[0].to_pydatetime() - MonthEnd(2)
    # Get last end of month + 1 day, as yahoo finance downloads up to but excluding 'end' date.
    end = pd.to_datetime(opt_ret["date"]).iloc[-1].to_pydatetime() + timedelta(days=1)
    sp500_prices = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, prepost=False)

    # Only consider end of month prices of SP500 data -> to calculate monthly returns.
    sp500_monthend = sp500_prices.iloc[sp500_prices.reset_index(drop=False).groupby(sp500_prices.index.to_period("M"))["Date"].idxmax()]
    # Bring all "exchange" eom dates  to the calendar end of months to align with dfs (opton return dates).
    sp500_monthend.index = sp500_monthend.index + MonthEnd(0)

    # Save to csv.
    sp500_monthend.to_csv(path/"sp500TR_prices.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Downloads SP500 Total Return Index prices"
    " from Yahoo Finance and saves to given path.")
     
    # Data path where 'sp500_opt_ret.parquet' resides, to download same time period.
    path = Path.cwd()/"data"
    
    parser.add_argument("--path", type=str, 
                default=path,
                help="Path where the option return file is located, to download "
                "monthly SP500 index prices that are aligned with the same time "
                "period.")

    args = parser.parse_args()

    download_sp500TR_prices(args)