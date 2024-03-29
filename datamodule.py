import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
import torch
from pathlib import Path
import pytorch_lightning as pl

from utils.preprocess import YearEndIndeces, binary_categorize, multi_categorize
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

class DataModule(pl.LightningDataModule):
    """Dataset Loader for Pytorch Lightning (Neural Network)."""
    def __init__(self,
                 path: str, # will be converted to Path in __init__
                 year_idx: int,
                 dataset: str,
                 batch_size: int,
                 init_train_length: int,
                 val_length: int,
                 test_length: int,
                #  start_val: str, 
                #  start_test: str,
                 label_fn: str,
                 custom_data: pd.DataFrame = None,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["path"])
        
        # If the data is provided at initialization, use it. (E.g. used in feature importance)
        if custom_data is not None:
            self.data = custom_data
        else:
            path = Path(path)
            self.data = load_data(path, dataset)

        # Get year train, val, test split indeces.
        splitter = YearEndIndeces(
                                self.data["date"], 
                                init_train_length=init_train_length, 
                                val_length=val_length,
                                test_length=test_length,
                                )
        eoy_indeces = list(splitter.generate_idx())
        self.eoy_train = eoy_indeces[year_idx]["train"]
        self.eoy_val = eoy_indeces[year_idx]["val"]
        self.eoy_test = eoy_indeces[year_idx]["test"]

        # Truncate data to only use current train, val and test.
        self.data = self.data.iloc[:self.eoy_test]
        assert len(self.data) == self.eoy_test, "length of data is not equal to eoy_test"
            
        # # feature engineer data
        # self.data = feature_engineer(self.data)
        
        # Get the y vector.
        self.y = self.data["option_ret"]
        # Classify returns (floats) into classes.
        if label_fn == "binary":
            self.y = self.y.apply(binary_categorize)
        elif label_fn == "multi3":
            self.y = self.y.apply(multi_categorize, classes=3)
        elif label_fn == "multi5":
            self.y = self.y.apply(multi_categorize, classes=5)
        else:
            raise ValueError("Specify label_fn as either 'binary' or 'multi'")
        # Get the features X.
        self.X = self.data.drop(["option_ret"], axis=1)
        
        # Save dates and drop it from X.
        self.dates = self.X["date"]
        self.X = self.X.drop(["date"], axis=1)
        
        # Convert X and y to torch tensors (for GPU).
        self.X = torch.from_numpy(self.X.values).float() #-> will be standardized in setup, so do it there.
        self.y = torch.from_numpy(self.y.values)
        
    def setup(self, stage: str = None):
        # Training data.
        # self.X_train = self.X[self.dates < self.hparams.start_val]
        self.X_train = self.X[:self.eoy_train]
        self.y_train = self.y[:len(self.X_train)]
        
        # Validation data.
        # mask = (self.dates >= self.hparams.start_val) & (self.dates < self.hparams.start_test)
        # self.X_val = self.X[mask]
        self.X_val = self.X[self.eoy_train:self.eoy_val]
        self.y_val = self.y[len(self.X_train):len(self.X_train)+len(self.X_val)]
        
        # Test data.
        self.X_test = self.X[self.eoy_val:self.eoy_test]
        self.y_test = self.y[-len(self.X_test):]
        
        assert (len(self.X_train)+len(self.X_val)+len(self.X_test)) == len(self.data), \
            "sum of X train, val, test is not equal length of dataset"
        assert (len(self.y_train)+len(self.y_val)+len(self.y_test) == len(self.data)), \
        "sum of y train, val, test is not equal to length of dataset"
        
        # Get mean and std of X_train.
        mean = torch.mean(self.X_train, axis=0)
        std = torch.std(self.X_train, axis=0)
        
        # Standardize X_train, X_val and X_test with mean/std from X_train.
        self.X_train = (self.X_train - mean) / std
        self.X_val = (self.X_val - mean) / std
        self.X_test = (self.X_test - mean) / std

        # Save important variables to pass to model class.
        # Input dim of features (int).
        self.input_dim = self.X_train.shape[1]
        # Number of classes (int).
        self.num_classes = len(self.y_train.unique())
        # Class weights (torch.tensor).
        self.class_weights = len(self.y_train) / self.y_train.unique(return_counts=True)[1]

        if self.hparams.custom_data is None:
            print("*****************************************************************************************")
            print("Current TORCH dataset information:")
            print("---")
            print("class counts: ", self.y_train.unique(return_counts=True))
            print("class_weights:", self.class_weights)
            print("device of class_weights:", self.class_weights.device)
            print("---")
            print(f"# of input data: {len(self.data)} with shape: {self.data.shape}")
            print(f"# of training samples: {len(self.y_train)} with X_train of shape: {self.X_train.shape}")
            print(f"# of validation samples: {len(self.y_val)} with X_val of shape: {self.X_val.shape}")
            print(f"# of test samples: {len(self.y_test)} with X_test of shape: {self.X_test.shape}")
            print("---")
            print(f"Train start date: ", self.dates.iloc[0].strftime("%Y-%m-%d"), 
                ", Train end date: ", self.dates.iloc[:self.eoy_train].iloc[-1].strftime("%Y-%m-%d"))
            print(f"Val start date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[0].strftime("%Y-%m-%d"), 
                ", Val end date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[-1].strftime("%Y-%m-%d"))
            print(f"Test start date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[0].strftime("%Y-%m-%d"), 
                ", Test end date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[-1].strftime("%Y-%m-%d"))
            print("*****************************************************************************************")
        else:
            print(f"Test start date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[0].strftime("%Y-%m-%d"), 
                ", Test end date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[-1].strftime("%Y-%m-%d"))



    def example(self):
        """Returns a random training example."""        
        idx = np.random.randint(0, len(self.X_train))
        x, y = self.X_train[idx], self.y_train[idx]
        return (x, y)

    def train_dataloader(self):
        dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                         num_workers=0, #uses just the main worker, see https://stackoverflow.com/questions/71713719/runtimeerror-dataloader-worker-pids-15876-2756-exited-unexpectedly
                         # there are issues occuring on windows where PID workers exit unexpectedly.
                         pin_memory=True,
                         shuffle=True, #shuffle training data
                         )

    def val_dataloader(self):
        dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                         num_workers=0,
                         pin_memory=True,
                         shuffle=False,
                         )

    def test_dataloader(self):
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                         num_workers=0,
                         pin_memory=True,
                         shuffle=False, #must not shuffle here!
                         )

    def predict_dataloader(self):
        dataset = self.X_test # predict_step expects tensor not a list
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                        num_workers=0,
                        pin_memory=True,
                        shuffle=False, #must not shuffle here!
                        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule for Lightning")
        parser.add_argument("--batch_size", type=int, default=512)
        return parent_parser


#****************************************************************************************

class Dataset():
    """Dataset for non-torch classifiers. Provides train, val and test set
    in numpy. Can also output predefinded cv splits for gridsearch."""
    def __init__(self, 
                path: str, 
                year_idx: int, 
                dataset: str, 
                init_train_length: int, 
                val_length: int,
                test_length: int,
                label_fn: str,
                custom_data: pd.DataFrame = None,
                ):
        
        # If the data is provided at initialization, use it. (E.g. used in feature importance)
        if custom_data is not None:
            self.data = custom_data
        else:
            path = Path(path)
            self.data = load_data(path, dataset)

        # Get year train, val, test split indeces.
        splitter = YearEndIndeces(
                                self.data["date"], 
                                init_train_length=init_train_length, 
                                val_length=val_length, 
                                test_length=test_length,
                                )
        eoy_indeces = list(splitter.generate_idx())
        self.eoy_train = eoy_indeces[year_idx]["train"]
        self.eoy_val = eoy_indeces[year_idx]["val"]
        self.eoy_test = eoy_indeces[year_idx]["test"]
        
        # Truncate data to only use train, val and current test.
        self.data = self.data.iloc[:self.eoy_test]
        assert len(self.data) == self.eoy_test, "length of data is not equal to eoy_test"
            
        # feature engineer data
        # self.data = feature_engineer(self.data)
        
        # Get the y vector.
        self.y = self.data["option_ret"]
        # Classify returns (floats) into classes.
        if label_fn == "binary":
            self.y = self.y.apply(binary_categorize)
        elif label_fn == "multi3":
            self.y = self.y.apply(multi_categorize, classes=3)
        elif label_fn == "multi5":
            self.y = self.y.apply(multi_categorize, classes=5)
        else:
            raise ValueError("Specify label_fn as either 'binary' or 'multi'")
        # Get the features X.
        self.X = self.data.drop(["option_ret"], axis=1)
        
        # Save dates and drop it from X.
        self.dates = self.X["date"]
        self.X = self.X.drop(["date"], axis=1)
        
#         # to torch Tensor
#         self.X = torch.from_numpy(self.X.values).float() #-> will be standardized in setup, so do it there.
#         self.y = torch.from_numpy(self.y.values)

        # Convert X and y to numpy arrays.
        self.X = self.X.values #-> will be standardized in setup, so do it there.
        self.y = self.y.values
    
        ############################### setup #########################################################
        # Training data.
        self.X_train = self.X[:self.eoy_train]
        self.y_train = self.y[:len(self.X_train)]
        
        # Validation data.
        self.X_val = self.X[self.eoy_train:self.eoy_val]
        self.y_val = self.y[len(self.X_train):len(self.X_train)+len(self.X_val)]
        
        # Test data.
        self.X_test = self.X[self.eoy_val:self.eoy_test]
        self.y_test = self.y[-len(self.X_test):]
        
        assert (len(self.X_train)+len(self.X_val)+len(self.X_test)) == len(self.data), \
            "sum of X train, val, test is not equal length of dataset"
        assert (len(self.y_train)+len(self.y_val)+len(self.y_test) == len(self.data)), \
        "sum of y train, val, test is not equal to length of dataset"

        
        # --> StandardScaler is instead used!
#         #standardize X_train
#         mean = torch.mean(self.X_train, axis=0)
#         std = torch.std(self.X_train, axis=0)
        
#         # Standardize X_train, X_val and X_test with mean/std from X_train
#         self.X_train = (self.X_train - mean) / std
#         self.X_val = (self.X_val - mean) / std
#         self.X_test = (self.X_test - mean) / std

        # Save variables
        # Input dim of features (int).
        self.input_dim = self.X_train.shape[1]
        # Number of classes (int).
        self.num_classes = len(np.unique(self.y_train))

        # Class weights (dict).
        # self.class_weights = len(self.y_train) / np.unique(self.y_train, return_counts=True)[1]
        # calculate "balanced" class weights manually (class_weight="balanced" not possible for TuneSearch)
        weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        labels = np.unique(self.y_train)
        self.class_weights = {}
        for i in range(len(labels)):
            self.class_weights[labels[i]] = weights[i]

        if custom_data is None:
            print("*****************************************************************************************")
            print("Current NUMPY dataset information:")
            print("---")
            print("class counts: ", np.unique(self.y_train, return_counts=True))
            print("class_weights:", self.class_weights)
            print("---")
            print(f"# of input data: {len(self.data)} with shape: {self.data.shape}")
            print(f"# of training samples: {len(self.y_train)} with X_train of shape: {self.X_train.shape}")
            print(f"# of validation samples: {len(self.y_val)} with X_val of shape: {self.X_val.shape}")
            print(f"# of test samples: {len(self.y_test)} with X_test of shape: {self.X_test.shape}")
            print("---")
            print(f"Train start date: ", self.dates.iloc[0].strftime("%Y-%m-%d"), 
                ", Train end date: ", self.dates.iloc[:self.eoy_train].iloc[-1].strftime("%Y-%m-%d"))
            print(f"Val start date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[0].strftime("%Y-%m-%d"), 
                ", Val end date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[-1].strftime("%Y-%m-%d"))
            print(f"Test start date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[0].strftime("%Y-%m-%d"), 
                ", Test end date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[-1].strftime("%Y-%m-%d"))
            print("*****************************************************************************************")
        else:
            print("*****************************************************************************************")
            print(f"Test start date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[0].strftime("%Y-%m-%d"), 
                ", Test end date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[-1].strftime("%Y-%m-%d"))
            print("*****************************************************************************************")
        
    def get_datasets(self):
        return self.X_train, self.X_val, self.X_test
    
    def get_cv_data(self):
        """For scikitlearn classifiers: return datasets and predefined cv split 
        needed for gridsearchcv (only 1 train/val split)"""
        # careful: if predicting on X_val later... -> cheating
        X = np.concatenate((self.X_train, self.X_val))
        y = np.concatenate((self.y_train, self.y_val))
        ps = PredefinedSplit(np.concatenate((np.zeros(len(self.X_train)) - 1, np.ones(len(self.X_val)))))
        
        assert (self.X_train.shape[0] + self.X_val.shape[0] == X.shape[0] and 
                (self.X_train.shape[1] == self.X_val.shape[1] == X.shape[1]))
        assert ps.get_n_splits() == 1, "There should only be 1 train/ val split in PredefinedSplit."
        
        return X, y, ps

    def get_train_val(self):
        """Used in xgboost trainer."""
        return self.X_train, self.X_val, self.y_train, self.y_val

    def get_test(self):
        return self.X_test, self.y_test

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset for Scikitlearn + xgboost")
        # parser.add_argument("--batch_size", type=int, default=512)
        return parent_parser


def load_data(path_data: Path, dataset: str):
    """Loads specific dataset from path, depending on specified size."""
    if dataset == "small":
        return pd.read_parquet(path_data/"final_df_call_cao_small.parquet")
    elif dataset == "medium":
        return pd.read_parquet(path_data/"final_df_call_cao_med_fillmean.parquet")
    elif dataset == "big":
        return pd.read_parquet(path_data/"final_df_call_cao_big_fillmean.parquet")
    else:
        raise ValueError("Specify dataset as either 'small', 'medium' or big'")