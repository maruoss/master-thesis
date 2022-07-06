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
                hidden_dim: int,
                learning_rate: float,
                config: dict = None,
        ):
        super().__init__()
        # Init variables are saved, so that model can be reloaded cleanly if necessary
        # self.save_hyperparameters(ignore=["class_weights"])
        self.save_hyperparameters()
        #ignore "dm" is crucial if dm is passed

        # tune or train?
        if config is not None:
            self.hidden_dim = config["hidden_dim"]
            self.learning_rate = config["lr"]
        else:
            self.hidden_dim = hidden_dim
            self.learning_rate = learning_rate
        
        
        #model
        self.l1 = nn.Linear(input_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, num_classes)
        
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
        num_classes=num_classes, average="macro") # should be equal to sklearn bal. acc.

        self.val_acc = torchmetrics.Accuracy()
        self.val_bal_acc= torchmetrics.Accuracy(
            num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
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
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FFN")
        parser.add_argument("--no_class_weights", action='store_true')
        parser.add_argument("--hidden_dim", type=int, default=100)
        parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)

        return parent_parser
        