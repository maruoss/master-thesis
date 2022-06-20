# %%
import numpy as np
import pandas as pd
import pathlib
import time
import matplotlib.pyplot as plt
from tqdm import trange

from preprocess import binary_categorize, feature_engineer


# %% 
# set path
path = pathlib.Path(r"C:\Users\Mathiass\OneDrive - Universität Zürich UZH\Documents\mt_literature")

# read dataset
data = pd.read_parquet(path/"final_df_filledmean_small.parquet")

# %%
# engineer option characteristics variables
data = feature_engineer(data)

# %%
# create X and y datasets
y = data["option_ret"]
# drop it for X
X = data.drop(["option_ret"], axis=1)

# %%
# apply label function to option returns
y = y.apply(binary_categorize)

# %%

# define start years for validation and test periods
# start_train = 1996 (fixed)
start_val = "2015"
start_test = "2016"

# extract and save dates column -> needed to sort into train, val, test
dates = X["date"]
X = X.drop(["date"], axis=1)

# Create train, val, test sets & convert to numpy
# Train
X_train = X[dates < start_val].values
y_train = y[:len(X_train)].values

# Val
mask = (dates >= start_val) & (dates < start_test)
X_val = X[mask].values
y_val = y[len(X_train):len(X_train)+len(X_val)].values

# Test
X_test = X[dates >= start_test].values
y_test = y[-len(X_test):].values
# %%

# Find end of year dates

eoy_array = np.where((data["date"].dt.year.diff() == 1))[0] - 1




# %%
eoy_array

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# %%





# %%

train_indeces = eoy_array[4:]
val_indeces = eoy_array[6:]

# %%

# class train_val_gen:
#     def __init__(self, length_train, length_val, length_test):
#         self.length_train = length_train
#         self.length_val = length_val
#         self.length_test = length_test

#     def generate(eoy_array):

# %%

def train_val_idx():
    for i in range(len(eoy_array) - 6):
        yield (list(range(train_indeces[i])), list(range(train_indeces[i], val_indeces[i])))

# %%

a = train_val_idx()


# %%
(next(a))
# %%
*_, last = a # for a better understanding check PEP 448
print(last)
# %%
last[0][-1]
# %%
item = None
for item in a:
    pass
# %%
