import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
import torch

from utils.preprocess import CVSplitter, binary_categorize, feature_engineer, multi_categorize
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import pathlib
import pdb
from sklearn.utils.class_weight import compute_class_weight

class MyDataModule_Loop(pl.LightningDataModule):
    def __init__(self,
                 path: str, # will be converted to Path in __init__
                 year_idx: int,
                 dataset: str,
                 batch_size: int,
                 init_train_length: int,
                 val_length: int,
                #  start_val: str, 
                #  start_test: str,
                 label_fn: str,
                #  config: dict = None,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["path"])
        
        # read data from disk
        path = pathlib.Path(path)
        if dataset == "small":
            self.data = pd.read_parquet(path/"final_df_filledmean_small.parquet")
        elif dataset == "big":
            self.data = pd.read_parquet(path/"final_df_filledmean.parquet")
        else:
            raise ValueError("Specify dataset as either 'small' or 'big'")

        # get splits
        splitter = CVSplitter(self.data["date"], init_train_length=init_train_length, 
                                val_length=val_length)
        eoy_indeces = list(splitter.generate_idx())
        self.eoy_train = eoy_indeces[year_idx]["train"]
        self.eoy_val = eoy_indeces[year_idx]["val"]
        self.eoy_test = eoy_indeces[year_idx]["test"]

        # Truncate data
        self.data = self.data.iloc[:self.eoy_test]
        assert len(self.data) == self.eoy_test, "length of data is not equal to eoy_test"
            
        # feature engineer data
        self.data = feature_engineer(self.data)
        
        # create y
        self.y = self.data["option_ret"]
        # make classification problem
        if label_fn == "binary":
            self.y = self.y.apply(binary_categorize)
        elif label_fn == "multi":
            self.y = self.y.apply(multi_categorize)
        else:
            raise ValueError("Specify label_fn as either 'binary' or 'multi'")
        # create X
        self.X = self.data.drop(["option_ret"], axis=1)
        
        # save dates and drop
        self.dates = self.X["date"]
        self.X = self.X.drop(["date"], axis=1)
        
        # to torch Tensor
        self.X = torch.from_numpy(self.X.values).float() #-> will be standardized in setup, so do it there.
        self.y = torch.from_numpy(self.y.values)
        
    def setup(self, stage: str = None):
        # train
        # self.X_train = self.X[self.dates < self.hparams.start_val]
        self.X_train = self.X[:self.eoy_train]
        self.y_train = self.y[:len(self.X_train)]
        
        #val
        # mask = (self.dates >= self.hparams.start_val) & (self.dates < self.hparams.start_test)
        # self.X_val = self.X[mask]
        self.X_val = self.X[self.eoy_train:self.eoy_val]
        self.y_val = self.y[len(self.X_train):len(self.X_train)+len(self.X_val)]
        
        # test
        self.X_test = self.X[self.eoy_val:self.eoy_test]
        self.y_test = self.y[-len(self.X_test):]
        
        assert (len(self.X_train)+len(self.X_val)+len(self.X_test)) == len(self.data), \
            "sum of X train, val, test is not equal length of dataset"
        assert (len(self.y_train)+len(self.y_val)+len(self.y_test) == len(self.data)), \
        "sum of y train, val, test is not equal to length of dataset"
        
        #standardize X_train
        mean = torch.mean(self.X_train, axis=0)
        std = torch.std(self.X_train, axis=0)
        
        # Standardize X_train, X_val and X_test with mean/std from X_train
        self.X_train = (self.X_train - mean) / std
        self.X_val = (self.X_val - mean) / std
        self.X_test = (self.X_test - mean) / std

        # Save variables to pass to model class
        # input dim
        self.input_dim = self.X_train.shape[1]
        # number of classes
        self.num_classes = len(self.y_train.unique())
        # class weights
        self.class_weights = len(self.y_train) / self.y_train.unique(return_counts=True)[1]

        print("*****************************************************************************************")
        print("Current TORCH dataset information:")
        print("---")
        print("class_weights:", self.class_weights)
        print("device of class_weights:", self.class_weights.device)
        print("---")
        print(f"# of input data: {len(self.data)} with shape: {self.data.shape}")
        print(f"# of training samples: {len(self.y_train)} with X_train of shape: {self.X_train.shape}")
        print(f"# of validation samples: {len(self.y_val)} with X_val of shape: {self.X_val.shape}")
        print(f"# of test samples: {len(self.y_test)} with X_test of shape: {self.X_test.shape}")
        print("---")
        print(f"train start date: ", self.dates.iloc[0].strftime("%Y-%m-%d"), 
              ", train end date: ", self.dates.iloc[:self.eoy_train].iloc[-1].strftime("%Y-%m-%d"))
        print(f"val start date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[0].strftime("%Y-%m-%d"), 
              ", val end date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[-1].strftime("%Y-%m-%d"))
        print(f"test start date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[0].strftime("%Y-%m-%d"), 
              ", test end date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[-1].strftime("%Y-%m-%d"))
        print("*****************************************************************************************")


    def example(self):
        """Returns a random training example."""        
        idx = np.random.randint(0, len(self.X_train))
        x, y = self.X_train[idx], self.y_train[idx]
        return (x, y)

    def train_dataloader(self):
        dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         )

    def val_dataloader(self):
        dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         )

    def test_dataloader(self):
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         )

    def predict_dataloader(self):
        dataset = self.X_test # predict_step expects tensor not a list
        return DataLoader(dataset, batch_size=len(self.X_test), #load the whole testset
                    num_workers=4,
                    pin_memory=True,
                    )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule for Lightning")
        parser.add_argument("--dataset", type=str, default="small")
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--label_fn", type=str, default="binary", 
                            choices=["binary", "multi"])

        return parent_parser


#****************************************************************************************

class Dataset():
    """Dataset non-torch classifiers. Only provides train, val and test set
    in numpy. Can also output predefinded cv splits for gridsearch."""
    def __init__(self, 
                path: str, 
                year_idx: int, 
                dataset: str, 
                init_train_length: int, 
                val_length: int, 
                label_fn: str,
                ):

        # read data from disk
        path = pathlib.Path(path)
        if dataset == "small":
            self.data = pd.read_parquet(path/"final_df_filledmean_small.parquet")
        elif dataset == "big":
            self.data = pd.read_parquet(path/"final_df_filledmean.parquet")
        else:
            raise ValueError("Specify dataset as either 'small' or 'big'")

        # get splits
        splitter = CVSplitter(self.data["date"], init_train_length=init_train_length, 
                                val_length=val_length, test_length=1)
        eoy_indeces = list(splitter.generate_idx())
        self.eoy_train = eoy_indeces[year_idx]["train"]
        self.eoy_val = eoy_indeces[year_idx]["val"]
        self.eoy_test = eoy_indeces[year_idx]["test"]
        
        # Truncate data
        self.data = self.data.iloc[:self.eoy_test]
        assert len(self.data) == self.eoy_test, "length of data is not equal to eoy_test"
            
        # feature engineer data
        self.data = feature_engineer(self.data)
        
        # create y
        self.y = self.data["option_ret"]
        # make classification problem
        if label_fn == "binary":
            self.y = self.y.apply(binary_categorize)
        elif label_fn == "multi":
            self.y = self.y.apply(multi_categorize)
        else:
            raise ValueError("Specify label_fn as either 'binary' or 'multi'")
        # create X
        self.X = self.data.drop(["option_ret"], axis=1)
        
        # save dates and drop
        self.dates = self.X["date"]
        self.X = self.X.drop(["date"], axis=1)
        
#         # to torch Tensor
#         self.X = torch.from_numpy(self.X.values).float() #-> will be standardized in setup, so do it there.
#         self.y = torch.from_numpy(self.y.values)

        # to numpy
        self.X = self.X.values #-> will be standardized in setup, so do it there.
        self.y = self.y.values
    
        ############################### setup #########################################################
        # train
        self.X_train = self.X[:self.eoy_train]
        self.y_train = self.y[:len(self.X_train)]
        
        #val
        self.X_val = self.X[self.eoy_train:self.eoy_val]
        self.y_val = self.y[len(self.X_train):len(self.X_train)+len(self.X_val)]
        
        # test
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
        # input dim
        self.input_dim = self.X_train.shape[1]
        # number of classes
        self.num_classes = len(np.unique(self.y_train))

        #class weights
        # self.class_weights = len(self.y_train) / np.unique(self.y_train, return_counts=True)[1]
        # calculate "balanced" class weights manually (class_weight="balanced" not possible for TuneSearch)
        weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        labels = np.unique(self.y_train)
        self.class_weights = {}
        for i in range(len(labels)):
            self.class_weights[labels[i]] = weights[i]

        print("*****************************************************************************************")
        print("Current NUMPY dataset information:")
        print("---")
        print("class_weights:", self.class_weights)
        print("---")
        print(f"# of input data: {len(self.data)} with shape: {self.data.shape}")
        print(f"# of training samples: {len(self.y_train)} with X_train of shape: {self.X_train.shape}")
        print(f"# of validation samples: {len(self.y_val)} with X_val of shape: {self.X_val.shape}")
        print(f"# of test samples: {len(self.y_test)} with X_test of shape: {self.X_test.shape}")
        print("---")
        print(f"train start date: ", self.dates.iloc[0].strftime("%Y-%m-%d"), 
              ", train end date: ", self.dates.iloc[:self.eoy_train].iloc[-1].strftime("%Y-%m-%d"))
        print(f"val start date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[0].strftime("%Y-%m-%d"), 
              ", val end date: ", self.dates.iloc[self.eoy_train:self.eoy_val].iloc[-1].strftime("%Y-%m-%d"))
        print(f"test start date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[0].strftime("%Y-%m-%d"), 
              ", test end date: ", self.dates.iloc[self.eoy_val:self.eoy_test].iloc[-1].strftime("%Y-%m-%d"))
        print("*****************************************************************************************")
        
    def get_datasets(self):
        return self.X_train, self.X_val, self.X_test
    
    def get_cv_data(self):
        """return datasets and cv split needed for gridsearchcv
        only 1 train/val split"""
        # careful: if predicting on X_val later... -> cheating
        X = np.concatenate((self.X_train, self.X_val))
        y = np.concatenate((self.y_train, self.y_val))
        ps = PredefinedSplit(np.concatenate((np.zeros(len(self.X_train)) - 1, np.ones(len(self.X_val)))))
        
        assert (self.X_train.shape[0] + self.X_val.shape[0] == X.shape[0] and (self.X_train.shape[1] == self.X_val.shape[1] == X.shape[1]))
        assert ps.get_n_splits() == 1, "more than one train/ val split in PredefinedSplit"
        
        return X, y, ps

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset for Scikitlearn + xgboost")
        parser.add_argument("--dataset", type=str, default="small")
        # parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--label_fn", type=str, default="binary", 
                            choices=["binary", "multi"])

        return parent_parser
