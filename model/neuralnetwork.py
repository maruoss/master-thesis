import time
import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F

# import pdb

class FFN(pl.LightningModule):
    def __init__(self,
                input_dim: int,
                num_classes: int,
                class_weights: torch.Tensor,
                no_class_weights: bool,
                learning_rate: float,
                hidden_dim: int,
                n_hidden: int,
                batch_norm: bool,
                dropout: bool,
                drop_prob: float,
                # config: dict = None,
        ):
        super().__init__()
        # Init variables are saved, so that model can be reloaded cleanly if necessary
        # self.save_hyperparameters(ignore=["class_weights"])
        self.save_hyperparameters()

        middle_layers = []
        for _ in range(self.hparams.n_hidden):
            middle_layers.append(nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim))
            if self.hparams.batch_norm:
                middle_layers.append(nn.BatchNorm1d(self.hparams.hidden_dim))
            middle_layers.append(nn.ReLU(inplace=True))
            if self.hparams.dropout:
                middle_layers.append(nn.Dropout(p=self.hparams.drop_prob))

        #model
        self.first = nn.Sequential(nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim), 
                                    nn.ReLU(inplace=True))
        self.middle = nn.Sequential(*middle_layers)  
        self.last = nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)
        
        #sample weights
        if not self.hparams.no_class_weights:
            self.class_weights = class_weights
            self.class_weights = self.class_weights.cuda() # Move to cuda, otherwise mismatch of devices # in train/val
        else:
            self.class_weights = None
        print("---")
        print("class_weights:", self.class_weights)
        print("device of class_weights:", self.class_weights.device)
        print("device of class:", self.device)
        print("---")

        #metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_bal_acc = torchmetrics.Accuracy(
        num_classes=self.hparams.num_classes, average="macro") # should be equal to sklearn bal. acc.

        self.val_acc = torchmetrics.Accuracy()
        self.val_bal_acc= torchmetrics.Accuracy(
            num_classes=self.hparams.num_classes, average="macro")

    def forward(self, x):
        x = self.first(x)
        x = self.middle(x)
        x = self.last(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #logits
        
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("loss/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        
        self.train_acc(y_hat, y)
        self.log("acc/train", self.train_acc, on_step=False, on_epoch=True)
        
        self.train_bal_acc(y_hat, y)
        self.log("bal_acc/train", self.train_bal_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #logits
        
#         self.log("hp_metric", torch.mean(y_hat.argmax(dim=-1).float()).item(), prog_bar=True) # average prediction class
        self.log("mean_pred", torch.mean(y_hat.argmax(dim=-1).float()).item(), prog_bar=True)
        
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("loss/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_acc(y_hat, y)
        self.log("acc/val", self.val_acc, on_step=False, on_epoch=True)
        
        self.val_bal_acc(y_hat, y)
        self.log("bal_acc/val", self.val_bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"val_loss": loss}
    
    def on_train_start(self):
        self.st_total = time.time()

    def on_train_epoch_start(self):
        self.st = time.time()
        self.steps = self.global_step

    def on_train_epoch_end(self):
        elapsed = time.time() - self.st
        steps_done = self.global_step - self.steps
        self.log("time/step", elapsed / steps_done)

    def on_train_end(self):
        elapsed = time.time() - self.st_total
        print(f"Total Training Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)

        self.log("loss/test_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        return self(batch)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FFN")
        parser.add_argument("--no_class_weights", action='store_true')
        parser.add_argument("--hidden_dim", type=int, default=100)
        parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
        parser.add_argument("--n_hidden", type=int, default=0)
        parser.add_argument("--no_batch_norm", action='store_false')
        parser.add_argument("--no_dropout", action='store_false')
        parser.add_argument("--drop_prob", type=float, default=0.5)

        return parent_parser
        