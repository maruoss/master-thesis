import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from pathlib import Path


def load_pfret_from_pfs(path_portfolios: Path) -> pd.DataFrame:
    """Load the 'option_ret' column of each (class and long-short) portfolio saved 
    in the 'path_portfolios' folder and concatenate to a dataframe.
        
        Args:
            path_data (Path):          Path where the portfolio .csv files reside.   

        Returns:
            pf_returns (pd.DataFrame): All option returns in columns in a dataframe.
    """
    pf_returns = []
    for file in Path.iterdir(path_portfolios):
        try:
            df = pd.read_csv(path_portfolios/file, parse_dates=["date"], index_col="date")
        except PermissionError as err:
            raise PermissionError("The 'portfolios' subfolder must not contain directories."
                                  "Please move or delete the subdirectory.") from err
            # from err necessary for chaining
        pf_returns.append(df["option_ret"].rename(file.name[:-4])) #rename Series to filename from portfolios.
    # Sort list in descending order first -> class0, class1, ..., etc.
    pf_returns = sorted(pf_returns, key=lambda x: x.name)
    pf_returns = pd.concat(pf_returns, axis=1) # Series names -> column names
    # Capitalize 'date' index name for plot axis label.
    pf_returns.index = pf_returns.index.rename("Date")
    return pf_returns


def load_rf_monthly(path_data: Path) -> pd.DataFrame:
    """Loads the monthly riskfree rate from the Fama French 5 factor dataset. 
    
    Citing from the file: "The 1-month TBill return is from Ibbotson and Associates Inc."
    """
    monthly_rf = pd.read_csv(path_data/"F-F_Research_Data_5_Factors_2x3.csv", skiprows=2, usecols=["Unnamed: 0", "RF"]).iloc[:706]
    # Rename Unnamed: 0 to Date and convert to appropriate Datetime format.
    monthly_rf = monthly_rf.rename(columns={"Unnamed: 0": "Date"})
    monthly_rf["Date"] = pd.to_datetime(monthly_rf["Date"], format="%Y%m") + MonthEnd(0)
    monthly_rf = monthly_rf.set_index("Date")
    # Convert ff columns to float (have been read as strings)
    monthly_rf["RF"] = monthly_rf["RF"].astype("float")
    # Divide by 100 (numbers are in percent)
    monthly_rf =  monthly_rf / 100
    return monthly_rf


def load_mkt_excess_ret_monthly(path_data: Path) -> pd.DataFrame:
    """Loads the monthly (market - riskfree) (excess) return from the Fama French 
    5 factor dataset.

    Citing Website: "[The return is composed of a]... value-weight return of all CRSP 
    firms incorporated in the US and listed on the NYSE, AM, or Nasdaq, ...)"""
    mkt_excess_ret_monthly = pd.read_csv(path_data/"F-F_Research_Data_5_Factors_2x3.csv", skiprows=2, usecols=["Unnamed: 0", "Mkt-RF"]).iloc[:706]
    # Rename Unnamed: 0 to Date and convert to appropriate Datetime format.
    mkt_excess_ret_monthly = mkt_excess_ret_monthly.rename(columns={"Unnamed: 0": "Date"})
    mkt_excess_ret_monthly["Date"] = pd.to_datetime(mkt_excess_ret_monthly["Date"], format="%Y%m") + MonthEnd(0)
    mkt_excess_ret_monthly = mkt_excess_ret_monthly.set_index("Date")
    # Convert ff columns to float (have been read as strings)
    mkt_excess_ret_monthly["Mkt-RF"] = mkt_excess_ret_monthly["Mkt-RF"].astype("float")
    # Divide by 100 (numbers are in percent)
    mkt_excess_ret_monthly =  mkt_excess_ret_monthly / 100
    return mkt_excess_ret_monthly


def load_ff_monthly(path_data: Path) -> pd.DataFrame:
    """Load the monthly data of the 5 Factors of Fama and French: Mkt-RF, SMB,
    HML, RMW, CMA and adjust dates to the true end of month dates.
        
        Args:
            path_data (Path):          Parent path where the Fama French data resides.   

        Returns:
            ff_monthly (pd.DataFrame): All factors in columns in a dataframe.
    """
    ff_monthly = pd.read_csv(path_data/"F-F_Research_Data_5_Factors_2x3.csv", skiprows=2).iloc[:706]
    # Rename Unnamed: 0 to Date and convert to appropriate Datetime format.
    ff_monthly = ff_monthly.rename(columns={"Unnamed: 0": "Date"})
    ff_monthly["Date"] = pd.to_datetime(ff_monthly["Date"], format="%Y%m") + MonthEnd(0)
    ff_monthly = ff_monthly.set_index("Date")
    # Convert ff columns to float (have been read as strings).
    for col in ff_monthly.columns:
        ff_monthly[col] = ff_monthly[col].astype("float")
    # Divide by 100 (numbers are in percent).
    ff_monthly =  ff_monthly / 100
    # Drop riskfree rate column (will not be used as a factor in the regressions).
    ff_monthly = ff_monthly.drop(columns="RF")
    return ff_monthly


def load_mom_monthly(path_data: Path) -> pd.DataFrame:
    """Load the monthly data of the Momentum factor from the Fama and French website
    and adjust dates to the true end of month dates.
        
        Args:
            path_data (Path):           Parent path where the Momentum .csv file resides.   

        Returns:
            mom_monthly (pd.DataFrame): The momentum factor in a column in a dataframe.
    """
    mom_monthly = pd.read_csv(path_data/"F-F_Momentum_Factor.csv", skiprows=13).iloc[:1144]
    # Rename Unnamed: 0 to Date and convert to appropriate Datetime format.
    mom_monthly = mom_monthly.rename(columns={"Unnamed: 0": "Date"})
    mom_monthly["Date"] = pd.to_datetime(mom_monthly["Date"], format="%Y%m") + MonthEnd(0)
    mom_monthly = mom_monthly.set_index("Date")
    # Strip whitespace of 'Mom    ' column and rename to all caps 'MOM'.
    mom_monthly = mom_monthly.rename(columns={mom_monthly.columns.item(): mom_monthly.columns.item().rstrip().upper()})
    # Convert Mom column to float (have been read as strings):
    mom_monthly["MOM"] =  mom_monthly["MOM"].astype("float")
    # Divide by 100 (numbers are in percent).
    mom_monthly =  mom_monthly / 100
    return mom_monthly

def load_vix_monthly(path_data: Path) -> pd.DataFrame:
    """Loads the daily data of the VIX factor from the CBOE website, select the
    end of month values and adjust dates to the true end of month dates.
       
       Args:
            path_data (Path):           Parent path where the VIX .csv file resides.   

        Returns:
            vix_monthly (pd.DataFrame): The momentum factor in a column in a dataframe.
    """
    vix = pd.read_csv(path_data/"VIX_History.csv", parse_dates=["DATE"])
    som_indeces = np.sort(np.concatenate([
                    np.where(vix["DATE"].dt.month.diff() == 1)[0],
                    np.where(vix["DATE"].dt.month.diff() == -11)[0]
                    ]))
    vix_monthly = vix.iloc[som_indeces - 1].rename(columns={"DATE": "Date"})
    vix_monthly = vix_monthly.set_index("Date")
    vix_monthly.index = vix_monthly.index + MonthEnd(0)
    vix_monthly = vix_monthly["CLOSE"].rename("VIX")
    return vix_monthly


def load_vvix_monthly(path_data: Path) -> pd.DataFrame:
    """Loads the daily data of the VVIX factor from the CBOE website, select the
    end of month values and adjust dates to the true end of month dates.
        
        Args:
            path_data (Path):           Parent path where the Fama French data resides.   

        Returns:
            mom_monthly (pd.DataFrame): The momentum factor in a column in a dataframe.
    """
    vvix = pd.read_csv(path_data/"VVIX_History.csv", parse_dates=["DATE"])
    som_indeces = np.sort(np.concatenate([
                    np.where(vvix["DATE"].dt.month.diff() == 1)[0],
                    np.where(vvix["DATE"].dt.month.diff() == -11)[0]
                    ]))
    vvix_monthly = vvix.iloc[som_indeces - 1].rename(columns={"DATE": "Date"})
    vvix_monthly = vvix_monthly.set_index("Date")
    vvix_monthly.index = vvix_monthly.index + MonthEnd(0)
    return vvix_monthly