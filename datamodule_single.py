import numpy as np
import pandas as pd
import torch

from utils.preprocess import binary_categorize, feature_engineer, multi_categorize
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

import pdb

class MyDataModule(pl.LightningDataModule):
    def __init__(self,
                 path,
                 dataset: str,
                 batch_size: int, 
                 start_val: str, 
                 start_test: str,
                 label_fn: str,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["path"])
        self.batch_size = batch_size
        
        # read data from disk
        if dataset == "small":
            self.data = pd.read_parquet(path/"final_df_filledmean_small.parquet")
        elif dataset == "big":
            self.data = pd.read_parquet(path/"final_df_filledmean.parquet")
        else:
            raise ValueError("Specify dataset as either 'small' or 'big'")
            
        # feature engineer data
        self.data = feature_engineer(self.data)
        
        # create y
        self.y = self.data["option_ret"]
        # make classification problem
        if label_fn == "binary":
            self.y = self.y.apply(binary_categorize)
        elif label_fn == "multi3":
            self.y = self.y.apply(multi_categorize, classes=3)
        elif label_fn == "multi5":
            self.y = self.y.apply(multi_categorize, classes=5)
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
        self.X_train = self.X[self.dates < self.hparams.start_val]
        self.y_train = self.y[:len(self.X_train)]
        
        #val
        mask = (self.dates >= self.hparams.start_val) & (self.dates < self.hparams.start_test)
        self.X_val = self.X[mask]
        self.y_val = self.y[len(self.X_train):len(self.X_train)+len(self.X_val)]
        
        # test
        self.X_test = self.X[self.dates >= self.hparams.start_test]
        self.y_test = self.y[-len(self.X_test):]
        
        assert (np.sum(len(self.X_train)+len(self.X_val)+len(self.X_test)) == len(self.data)), "sum of samples of splits\
        is not equal length of dataset"
        
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

        print("class_weights:", self.class_weights)
        print("device of class_weights:", self.class_weights.device)
        print("---")
        print(f"# of input data: {len(self.data)} with shape: {self.data.shape}")
        print(f"# of training samples: {len(self.y_train)} with X_train of shape: {self.X_train.shape}")
        print(f"# of validation samples: {len(self.y_val)} with X_val of shape: {self.X_val.shape}")
        print(f"# of test samples: {len(self.y_test)} with X_test of shape: {self.X_test.shape}")
        print(f"train start date: ", self.dates[self.dates < self.hparams.start_val].iloc[0].strftime("%Y-%m-%d"), 
              ", train end date: ", self.dates[self.dates < self.hparams.start_val].iloc[-1].strftime("%Y-%m-%d"))
        print(f"val start date: ", self.dates[mask].iloc[0].strftime("%Y-%m-%d"), 
              ", val end date: ", self.dates[mask].iloc[-1].strftime("%Y-%m-%d"))
        print(f"test start date: ", self.dates[self.dates >= self.hparams.start_test].iloc[0].strftime("%Y-%m-%d"), 
              ", test end date: ", self.dates[self.dates >= self.hparams.start_test].iloc[-1].strftime("%Y-%m-%d"))
        print("---")

    def example(self):
        """Returns a random training example."""        
        idx = np.random.randint(0, len(self.X_train))
        x, y = self.X_train[idx], self.y_train[idx]
        return (x, y)

    def train_dataloader(self):
        dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(dataset, batch_size=self.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         )

    def val_dataloader(self):
        dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(dataset, batch_size=self.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         )

    def test_dataloader(self):
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(dataset, batch_size=self.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--dataset", type=str, default="small")
        # parser.add_argument("--path", type=str, help="path to folder that contains the data")
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--start_val", type=str, default="2014")
        parser.add_argument("--start_test", type=str, default="2016")
        parser.add_argument("--label_fn", type=str, default="binary")

        return parent_parser
