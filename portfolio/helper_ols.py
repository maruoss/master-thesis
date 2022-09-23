from pathlib import Path
import numpy as np
import pandas as pd

import statsmodels.api as sm
from portfolio.stargazer import Stargazer


def regress_factors(regression_map: dict, factors_avail: pd.DataFrame, y: pd.Series, path_results: Path) -> None:
    """Performs each relevant regression from the 'regression_map' dictionary and 
    saves results as .txt (latex) and .csv files. in an accordingly named folder
    in 'path_results'. 

        Args:
            factors:        All independent variables concatenated in a Dataframe.
            y:              The dependent variable (long short monthly portfolio returns here).
            path_results:   The path where the results folder resides.
    
    """
    # Save every single regression in its own folder.
    collect_group = {}
    for group in regression_map.keys():
        group_map = regression_map[group]
        for regr_key in group_map.keys():
            path = path_results/group/regr_key
            path.mkdir(exist_ok=True, parents=True) #create group parent folder if it doesnt exist.
            X = factors_avail.loc[:, group_map[regr_key]]
            # Add constant for intercept.
            X = sm.add_constant(X)

            # 1a. Regression. No HAC Standard Errors.
            ols = sm.OLS(y, X) #long_short_return regressed on X.
            ols_result = ols.fit()
            fileprefix = "Standard"
            save_ols_results(ols_result, path, fileprefix)
            # Append to dict for stargazer summary.
            collect_group.setdefault(group, {}).setdefault(fileprefix, []).append(ols_result)

            # 1b. Regression. With HAC Standard Errors. Lags of Greene (2012): L = T**(1/4).
            max_lags = int(len(X)**(1/4))
            ols_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}, use_t=True)
            fileprefix = f"HAC_{max_lags}"
            save_ols_results(ols_result, path, fileprefix)
            # Append to dict for stargazer summary.
            collect_group.setdefault(group, {}).setdefault(fileprefix, []).append(ols_result)

            # 1c. Regression. With HAC Standard Errors. Lag after Bali (2021) = 12.
            ols_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": 12}, use_t=True)
            fileprefix = "HAC_12"
            save_ols_results(ols_result, path, fileprefix)
            # Append to dict for stargazer summary.
            collect_group.setdefault(group, {}).setdefault(fileprefix, []).append(ols_result)
    
    # Save results in stargazer format (group regressions side-by-side).
    for group in collect_group.keys():
        for cov_type in collect_group[group].keys():
            stargazer = Stargazer(collect_group[group][cov_type])
            # Model names.
            model_names = sorted(list(regression_map[group].keys()))
            stargazer.custom_columns(model_names, [1]*len(model_names))
            # Covariate order.
            # Largest list in 'group' determines covariate order in latex output.
            largest_list_group = max(list(regression_map[group].values()), key=lambda ls: len(ls))
            cov_order =  largest_list_group + ["const"] #add constant variable.
            stargazer.covariate_order(cov_order)
            # Add cov type in notes.
            stargazer.add_custom_notes([f"Cov. Type:\t{cov_type}"])
            with (path_results/group/f"stargazer_latex_{cov_type}.txt").open("w") as text_file:
                text_file.write(stargazer.render_latex(escape=True))
            with (path_results/group/f"stargazer_{cov_type}.html").open("w") as text_file:
                text_file.write(stargazer.render_html())


def save_ols_results(ols_result, path: Path, fileprefix: str) -> None:
    """Saves results from statsmodels.OLS to .txt and .csv files in a separate 
    folder in the path.

        Args:
            ols_result:             Object from ols.fit().summary()
            path (Path):            Path to save the files.
            fileprefix (str):       Prefix name of the files (e.g. cov type).
    
    """
    # LaTeX file.
    # NOTE: summary() doesnt work with adding significance stars + layout is not as nice.
    with (path/f"{fileprefix}_latex.txt").open("w") as text_file:        
        text_file.write("OLS Summary2 as LaTeX:\n\n")
        summary2 = ols_result.summary2(alpha=0.05)
        summary2.tables[1] = add_signif_stars_df(summary2.tables[1])
        summary2.tables[0] = replace_scale_with_cov_type(summary2.tables[0], 
                                                        fileprefix)
        text_file.write(summary2.as_latex())
        text_file.write("\n\n\nOLS Summary as LaTeX:\n\n")
        text_file.write(ols_result.summary(alpha=0.05).as_latex())
    # .txt file.
    with (path/f"{fileprefix}.txt").open("w") as text_file:
        text_file.write(summary2.as_text())
    # 1 .csv file.
    with (path/f"{fileprefix}_summary1.csv").open("w") as text_file:
        text_file.write(ols_result.summary(alpha=0.05).as_csv())
    # 2. .csv file
    with (path/f"{fileprefix}_summary2.csv").open("w") as text_file:
        for idx, df in enumerate(summary2.tables):
            # Only print rows and cols for table1, where they are annotated.
            if idx==1: 
                df.to_csv(text_file, line_terminator="\n")
            else:
                df.to_csv(text_file, line_terminator="\n", index=False, header=False)
            # text_file.write("\n")


def regress_on_constant(y: pd.Series, save_ols=False, save_ols_path: Path = None) -> dict:
    """Perform OLS with standard and two different max_lags for HAC Standard errors.

    Regress y on a constant -> Test the signficance of the mean being different from 0.
    Additionally, add significance stars to the statsmodels summary2 output.

    Args:
        y (pd.Series):                      Target array/ Series that is regressed 
                                            on the intercept.
        save_ols (Bool), optional:          Whether to save ols result in 
                                            'save_ols_path' as LaTeX, .txt and 
                                            .csv. Default: False.
        save_ols_path (Path), optional:     Path to the directory where ols 
                                            results are to be saved if 
                                            save_ols=True. Default: None.
    
    Returns:
        Only the tables[1] of the statsmodels.summary2() results.
    """
    if save_ols:
        if save_ols_path is None:
            raise ValueError("If 'save_ols' is True, 'save_ols_path' must be provided.")
        # Create folder if it doesnt exist yet.
        save_ols_path.mkdir(parents=False, exist_ok=True)
    # Create X, just 1's for the intercept.
    X = np.ones_like(y)
    ols_results = {}
    # 1a) No HAC Standard Errors.
    cov_type = "Standard"
    ols = sm.OLS(y, X) #long_short_return regressed on X.
    result_fit = ols.fit()
    if save_ols:
        save_ols_results(result_fit, save_ols_path, fileprefix=cov_type)
    table1 = result_fit.summary2(alpha=0.05).tables[1] #df
    table1 = add_signif_stars_df(table1)
    ols_results[cov_type] = table1
    # 1b) robust HAC Standard Errors. Lags of Greene (2012): L = T**(1/4).
    max_lags = int(len(X)**(1/4))
    cov_type = f"HAC_{max_lags}"
    result_fit = ols.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}, use_t=True)
    if save_ols:
        save_ols_results(result_fit, save_ols_path, fileprefix=cov_type)
    table1 = result_fit.summary2(alpha=0.05).tables[1] #df
    table1 = add_signif_stars_df(table1)
    ols_results[cov_type] = table1
    # 1c) robust HAC Standard Errors. Lag after Bali (2021) = 12.
    cov_type = "HAC_12"
    result_fit = ols.fit(cov_type="HAC", cov_kwds={"maxlags": 12}, use_t=True)
    if save_ols:
        save_ols_results(result_fit, save_ols_path, fileprefix=cov_type)
    table1 = result_fit.summary2(alpha=0.05).tables[1] #df
    table1 = add_signif_stars_df(table1)
    ols_results[cov_type] = table1
    return ols_results


def regress_on_X(y: pd.Series, X: np.array, save_ols=False, save_ols_path: Path = None) -> dict:
    """Perform OLS with standard and two different max_lags for HAC Standard errors.

    Regress y on X -> Test the signficance of the factors (+constant) in X.
    Additionally, add significance stars to the statsmodels summary2 output.

    Args:
        y (pd.Series):                      Target array/ Series that is regressed 
                                            on X.
        X (pd.DataFrame):                   Features/ Independent variables to
                                            examine for significance.
        save_ols (Bool), optional:          Whether to save ols result in 
                                            'save_ols_path' as LaTeX, .txt and 
                                            .csv. Default: False.
        save_ols_path (Path), optional:     Path to the directory where ols 
                                            results are to be saved if 
                                            save_ols=True. Default: None.
    
    Returns:
        Only the tables[1] of the statsmodels.summary2() results.
    """
    if save_ols:
        if save_ols_path is None:
            raise ValueError("If 'save_ols' is True, 'save_ols_path' must be provided.")
        # Create folder if it doesnt exist yet.
        save_ols_path.mkdir(parents=False, exist_ok=True)
    # Create X, just 1's for the intercept.
    ols_results = {}
    # 1a) No HAC Standard Errors.
    cov_type = "Standard"
    ols = sm.OLS(y, X)
    result_fit = ols.fit()
    if save_ols:
        save_ols_results(result_fit, save_ols_path, fileprefix=cov_type)
    table1 = result_fit.summary2(alpha=0.05).tables[1] #df
    table1 = add_signif_stars_df(table1)
    ols_results[cov_type] = table1
    # 1b) robust HAC Standard Errors. Lags of Greene (2012): L = T**(1/4).
    max_lags = int(len(X)**(1/4))
    cov_type = f"HAC_{max_lags}"
    result_fit = ols.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}, use_t=True)
    if save_ols:
        save_ols_results(result_fit, save_ols_path, fileprefix=cov_type)
    table1 = result_fit.summary2(alpha=0.05).tables[1] #df
    table1 = add_signif_stars_df(table1)
    ols_results[cov_type] = table1
    # 1c) robust HAC Standard Errors. Lag after Bali (2021) = 12.
    cov_type = "HAC_12"
    result_fit = ols.fit(cov_type="HAC", cov_kwds={"maxlags": 12}, use_t=True)
    if save_ols:
        save_ols_results(result_fit, save_ols_path, fileprefix=cov_type)
    table1 = result_fit.summary2(alpha=0.05).tables[1] #df
    table1 = add_signif_stars_df(table1)
    ols_results[cov_type] = table1
    return ols_results


def replace_scale_with_cov_type(ols_summary2_table0: pd.DataFrame, 
                                cov_type: str
                                ) -> pd.DataFrame:
    """Summary2 does not have the cov type by default in its tables. Here we replace
    the 'Scale' statistic at location (6, 2) and (6, 3) with the covariance type."""
    ols_summary2_table0.iloc[6, 2] = "Cov. Type"
    ols_summary2_table0.iloc[6, 3] = cov_type
    return ols_summary2_table0



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


def get_save_mean_significance(pf_returns: pd.DataFrame, parent_path: Path):
    save_ols_path = parent_path/"Mean_Significance"
    save_ols_path.mkdir(parents=False, exist_ok=True)
    # Assuming 3 standard error types are to be saved in the results.
    mean_signif_df = pd.DataFrame(np.zeros((pf_returns.shape[1], 3))*np.nan, dtype=str)
    mean_signif_df.index = pf_returns.columns
    for pf in list(pf_returns.columns):
        mean_signif_results = regress_on_constant(
                                pf_returns[pf], 
                                save_ols=True, #Save the detailed results in a separate folder as well.
                                save_ols_path=save_ols_path/pf,
                                )
        # Check means.
        for se in mean_signif_results.values():
            assert abs(se["Coef."].item() - pf_returns.mean(axis=0)[pf]) < 0.0000001
        # Add to final output series.
        se_significances = []
        for se in mean_signif_results.keys():
            se_significances.append(mean_signif_results[se]["Signif."].item())
        col_names_se = list(mean_signif_results.keys())
        mean_signif_df.loc[pf] = se_significances
        mean_signif_df.columns = col_names_se
    return mean_signif_df


def get_save_alphabeta_significance(pf_returns: pd.DataFrame,
                                    X: np.ndarray,
                                    parent_path: Path,
                                    alphas, #monthly alphas
                                    betas,
                                    ):
    # Assuming 3 standard error types are to be saved for alpha and beta. (3*2=6)
    alpha_signif_df = pd.DataFrame(np.zeros((pf_returns.shape[1], 3))*np.nan, dtype=str)
    beta_signif_df = pd.DataFrame(np.zeros((pf_returns.shape[1], 3))*np.nan, dtype=str)
    alpha_signif_df.index = pf_returns.columns
    beta_signif_df.index = pf_returns.columns

    save_ols_path = parent_path/"AlphaBeta_Significance"
    save_ols_path.mkdir(parents=False, exist_ok=True)
    for pf in list(pf_returns.columns):
        y = pf_returns[pf]
        alpha_beta_ols_results = regress_on_X(y, X, save_ols=True, 
                                                    save_ols_path=save_ols_path/pf)
        #*** Sanity Check
        for se in alpha_beta_ols_results.values(): #se for standard error.
            # Check if alpha/beta coincides with already calculated alpha/beta.
            assert abs(se["Coef."].iloc[0] - alphas[pf]) < 0.0000001
            assert abs(se["Coef."].iloc[1] - betas[pf]) < 0.0000001
        #***
        se_significances_alpha = []
        se_significances_beta = []
        for se in alpha_beta_ols_results.values():
            se_significances_alpha.append(se["Signif."].iloc[0])
            se_significances_beta.append(se["Signif."].iloc[1])
        se_names = list(alpha_beta_ols_results.keys())
        # Alpha.
        alpha_signif_df.columns = se_names
        alpha_signif_df.loc[pf] = se_significances_alpha
        # Beta.
        beta_signif_df.columns = se_names
        beta_signif_df.loc[pf] = se_significances_beta
    return alpha_signif_df, beta_signif_df