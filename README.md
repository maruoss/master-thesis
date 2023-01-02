# Option Return Classification with Machine Learning

This is the github repo for my master thesis submitted in partial fulfillment of the requirements for the degree of Master of Science in Quantitative Finance at ETH Zurich and University of Zurich.

---

<center> 

<b>Option Return Classification with Machine Learning</b>

Supervisor: Prof. Dr. Erich Walter Farkas

Co-Supervisor: Patrick Lucescu

Date of Submission: 30 November 2022

</center>

---

[Abstract](./abstract.pdf)

For questions contact me via mail: mruoss@student.ethz.ch.

---
## Running the code:
---

### Requirements
To run my code, please install the following dependencies in a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) environment:

```
conda env create -f environment.yml
```

### Training the models

The following data has to be added to the ***./data*** subfolder in order to train the ML models:

1.
    - small feature set (only option features): *final_df_call_cao_small.parquet*
    - medium feature set (+94 additional stock features of Green (2017): *final_df_call_cao_med_fillmean.parquet*
    - big feature set (+additional SIC codes): *final_df_call_cao_big_fillmean.parquet*

2. Then ***main.py*** can be run with the command:

```
python main.py tune [lin, rf, xgb, nn, tf]
```

where lin = logistic regression, rf = random forests, xgb = gradient boosting trees, nn = simple feedforward neural network, tf = transformer (encoder).

---
### Evaluate the portfolios formed with the predictions

1. The following data has to be added first to the ***/.data*** subfolder:

    - secid.csv (security id's of the underlying stocks of the options)
    - F-F_Research_Data_5_Factors_2x3.csv (5 factors of Fama and French (2015), Source: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research)
    - F-F_Momentum_Factorc.csv (Momentum factor of Carhart (1997), Source: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research)
    - VIX_History.csv (monthly levels from CBOE https://www.cboe.com/tradable_products/vix/vix_historical_data/)
    - VVIX_History.csv (monthly levels from CBOE https://www.cboe.com/tradable_products/vix/vix_historical_data/)

2. Then ***main_pf.py*** is first run with the argument *agg* to aggregate the predictions into portfolios:
    ```
    python main_pf.py agg [experiment id (foldername of experiment)]
    ```
3. Then performance evaluation can be done on these portfolios:

    ```
    python main_pf.py [perf, reg_pfs] [experiment id (foldername of experiment)]
    ```

4. Feature importances is then done with:

    ```
    python main_pf.py importance [experiment id (foldername of experiment)]
    ```