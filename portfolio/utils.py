import pandas as pd


def add_signif_stars_df(ols_summary2_tables1_df: pd.DataFrame) -> pd.DataFrame:
    # pd.apply handles both 1 variable df's and multi variable dfs...
    p_val = ols_summary2_tables1_df["P>|t|"] #pd.Series
    ols_summary2_tables1_df["Signif."] = p_val.apply(gen_stars) #create new column.
    return ols_summary2_tables1_df


def gen_stars(p_value) -> str:
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.01:
        return "*"
    else:
        return ""
