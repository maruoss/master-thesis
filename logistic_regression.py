# %%
import numpy as np
import pandas as pd
import pathlib
import time
import matplotlib.pyplot as plt
from tqdm import trange

from utils.preprocess import CVSplitter, binary_categorize, feature_engineer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# %% 
# set pathmain.py
path = pathlib.Path(r"C:\Users\Mathiass\OneDrive - Universität Zürich UZH\Documents\mt_literature\data")

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
start_val = "2014"
start_test = "2016"

# extract and save dates column -> needed to sort into train, val, test
dates = X["date"]
X = X.drop(["date"], axis=1)

# Create train, val, test sets & convert to numpy
# Train
train_dates = dates < start_val
X_train = X[train_dates].values
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

# eoy_array = np.where((data["date"].dt.year.diff() == 1))[0] - 1

# # %%
# eoy_array

# %%

# %%


# %%
# Create CV splitter and convert X_train to numpy
cv_split = CVSplitter(dates[train_dates], init_train_length=5, val_length=2)

a = cv_split.generate()

for train, test in a:
    print("train start: %s train end: %s :: test start: %s, test end: %s" 
    % (dates[train[0]].strftime("%Y-%m-%d"), dates[train[-1]].strftime("%Y-%m-%d"), 
    dates[test[0]].strftime("%Y-%m-%d"), dates[test[-1]].strftime("%Y-%m-%d"))) 

# %%

cv_split.train_eoy
# %%

output = cv_split.generate_idx()
# %%
for i in output:
    print(i)

#%%

output


# %%
scaler = StandardScaler()

# clf = LogisticRegression(random_state=0, 
#                          class_weight="balanced",
#                         #  solver="saga",
#                         #    max_iter=1000,
#                         #  n_jobs=-1, #slower if activated
#                         #  C=1e-12,
#                         # l1_ratio=0.5,
#                         )

# clf = RandomForestClassifier(random_state=0,
#                     class_weight="balanced",
#                     # max_depth=2,
#                     n_jobs=-1,
#                 )

clf = HistGradientBoostingClassifier(random_state=0,
                                 max_iter=100000, 
                                 max_depth=2,
#                                  learning_rate=1.0,
                                validation_fraction=None,
                                #  verbose=1,
                                )

# clf = LinearSVC(random_state=0,
#                class_weight="balanced",
#                )

# clf = SVC(random_state=0,
#          class_weight="balanced",
#          )


clf = Pipeline([
    # ('scaler', scaler),
    ('clf', clf)
])

# %%
param_grid = {
        #   "clf__C":  np.logspace(-5, 5, 5),
      "clf__max_depth": [1, 2, 3, 4, 5],
}
clf = GridSearchCV(
    clf, 
    param_grid,
    scoring=["accuracy", "balanced_accuracy"],
    refit="balanced_accuracy",
    n_jobs=-1,
    cv=cv_split.generate(),
    verbose=3,
)


# %%
############################### FIT ################################################
s_time = time.time()
clf.fit(X_train, y_train,
#        clf__sample_weight=sample_weight
       )
e_time = time.time()
print(f"Time to fit: {divmod(e_time - s_time, 60)[0]:.0f}:{divmod(e_time - s_time, 60)[1]:.0f}\
 min")

###################################################################################
# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

### 

print("y_val mean: ", np.mean(y_val))

### TRIVIAL
y_trivial = np.zeros_like(y_val)
print("trivial accuracy: ", accuracy_score(y_val, y_trivial))
print("trivial balanced accuracy: ", balanced_accuracy_score(y_val, y_trivial))

# %%
### CLASSIFIER
y_pred = clf.predict(X_val)

print("classfier accuracy: ", accuracy_score(y_val, y_pred))
print("classifier balanced accuracy: ", balanced_accuracy_score(y_val, y_pred))
print("prediction counts:", pd.value_counts(y_pred), sep="\n")
print("mean prediction: ", np.mean(y_pred))



#%%
#### DUMMY
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_dummy = dummy_clf.predict(X_val)


print("dummy accuracy: ", accuracy_score(y_val, y_dummy))
print("dummy balanced accuracy: ", balanced_accuracy_score(y_val, y_dummy))

# %%
# classfier accuracy:  0.5644384579255092
# classifier balanced accuracy:  0.5230255282223188
# prediction counts:
# 0    247167
# 1    123295
# dtype: int64
# mean prediction:  0.3328141617763765
# %%
clf.cv_results_

# %%

clf.best_params_
# %% To open tensorboard: paste below line in cmd
# tensorboard --logdir=logs --port=6007   