
# Manually add sic2 codes to dataframe. Only affects the 'big' dataset.
def add_sic_manually(df):
    """Adds missing sic codes to all rows with the respective permno number."""
    df.loc[df["permno"] == 18312, "sic2"] = 28 #Moderna https://sec.report/CIK/0001682852
    df.loc[df["permno"] == 18428, "sic2"] = 28 #Dow Inc https://sec.report/Ticker/DOW
    df.loc[df["permno"] == 93429, "sic2"] = 62 #CBOE Global Markets https://sec.report/Ticker/CBOE
    df.loc[df["permno"] == 18143, "sic2"] = 28 #Linde https://sec.report/CIK/0001707925
    df.loc[df["permno"] == 19285, "sic2"] = 35 #Carrier Global Corp https://sec.report/Ticker/CARR
    df.loc[df["permno"] == 18592, "sic2"] = 1 #Corteva https://sec.report/CIK/0001755672
    df.loc[df["permno"] == 17942, "sic2"] = 20 #Keurig Dr Pepper https://sec.report/Ticker/kdp
    df.loc[df["permno"] == 18420, "sic2"] = 48 #FOX Corp https://sec.report/Ticker/foxa
    df.loc[df["permno"] == 20057, "sic2"] = 28 #Viatris https://sec.report/Ticker/foxa
    df.loc[df["permno"] == 15850, "sic2"] = 72 #Match Group https://sec.report/Ticker/mtch
    df.loc[df["permno"] == 32791, "sic2"] = 35 #Weatherford https://sec.report/Ticker/wft
    df.loc[df["permno"] == 19286, "sic2"] = 36 #OTIS https://sec.report/Ticker/OTIS
    df.loc[df["permno"] == 17700, "sic2"] = 73 #Ceridian https://sec.report/Ticker/CDAY
    df.loc[df["permno"] == 18724, "sic2"] = 39 #Amcor https://sec.report/Ticker/AMCR
    df.loc[df["permno"] == 38850, "sic2"] = 24 #Skyline https://sec.report/Ticker/sky
    df.loc[df["permno"] == 19807, "sic2"] = 38 #Vontier https://sec.report/Ticker/VNT
    df.loc[df["permno"] == 78840, "sic2"] = 48 #IAC Interactive Corp ->sp500 const. list
    # ----
    df.loc[df["permno"] == 81594, "sic2"] = 13 #Dynegy, only 8 were NaN, rest was =13
    df.loc[df["permno"] == 88601, "sic2"] = 49 #Mirant, first 4 were NaN, rest =49
    df.loc[df["permno"] == 80913, "sic2"] = 73 #ACS, last month NaN, rest =73
    df.loc[df["permno"] == 44652, "sic2"] = 54 #American Stores Company, last month NaN, rest =54
    df.loc[df["permno"] == 76563, "sic2"] = 54 #Meyer Fred Inc., last month NaN, rest =54

    return df
