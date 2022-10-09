import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import time
import torchmetrics
import pytorch_lightning as pl


# based on https://colab.research.google.com/github/leox1v/dl20/blob/master/Transformers_Solution.ipynb#scrollTo=7JaF6C3Dfdog&uniqifier=2 
# and https://github.com/maruoss/deeprl-trading/blob/main/agents/transformer.py


class TransformerEncoder(pl.LightningModule):
    """Stacked Transformer Encoder Blocks for Return Classification.
    
    Args:
        d (int): The embedding dimension (for our data, no embedding, d=1).
        heads (int): The number of attention heads for each transformer block. Transformer paper: 8.
        depth (int): The number of transformer blocks. Transformer paper: 6.
        n_mlp (int): n_mlp * d is the hidden dim of the indep. FFNs in Transformer Block.
    """
    def __init__(self,
                input_dim: int, # "input sequence": l
                d: int, #embedding dimension: d
                num_classes: int,
                class_weights: torch.Tensor,
                no_class_weights: bool,
                depth: int, #The number of transformer blocks.
                heads:int, #The number of attention heads for each transformer block
                n_mlp: int, #n_mlp * d = hidden dim of independent FFNs
                learning_rate: float,
                # hidden_dim: int,
                # n_hidden: int,
                # batch_norm: bool,
                # dropout: bool,
                # drop_prob: float,
                # config: dict = None,
        ):
        super().__init__()
        # Init variables are saved, so that model can be reloaded cleanly if necessary
        # self.save_hyperparameters(ignore=["class_weights"])
        self.save_hyperparameters()

        #Embedding
        self.first = nn.Linear(1, d)

        #model
        # The stacked transformer blocks.
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d=d, heads=heads, n_mlp=n_mlp) for _ in range(depth)]
        )

        #De-Embedding, convert feature dimension d back to 1.
        self.seclast = nn.Linear(d, 1)
        # tensors of shape [b, l, d] -> assuming d=1 -> squeeze -> nn.Linear -> [b, num_classes]
        self.last = nn.Linear(input_dim, self.hparams.num_classes)
        
        #sample weights
        if not self.hparams.no_class_weights:
            self.class_weights = class_weights
            self.class_weights = self.class_weights.cuda() # Move to cuda, otherwise mismatch of devices # in train/val
        else:
            self.class_weights = None
        # print("---")
        # print("class_weights:", self.class_weights)
        # print("device of class_weights:", self.class_weights.device)
        # print("device of class:", self.device)
        # print("---")

        #metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_bal_acc = torchmetrics.Accuracy(
        num_classes=self.hparams.num_classes, average="macro") # should be equal to sklearn bal. acc.

        self.val_acc = torchmetrics.Accuracy()
        self.val_bal_acc= torchmetrics.Accuracy(
            num_classes=self.hparams.num_classes, average="macro")

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(dim=-1) #[b, l, 1]
        else:
            raise ValueError("Dimension of each feature has to be 1 before embedding them.")
        x = self.first(x) #[b, l, d]
        x = self.transformer_blocks(x) #[b, l, d]
        x = self.seclast(x) #[b, l, 1]
        x = x.squeeze(dim=-1) #[b, l]
        x = self.last(x) #[b, num_classes]
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #logits
        
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        # Logging is done "log_every_n_steps" times (default=50 steps)
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
        # parser.add_argument("--hidden_dim", type=int, default=100)
        parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
        # parser.add_argument("--heads", type=int, default=8)
        # parser.add_argument("--depth", type=int, default=6)
        # parser.add_argument("--n_hidden", type=int, default=0)
        # parser.add_argument("--no_batch_norm", action='store_false')
        # parser.add_argument("--no_dropout", action='store_false')
        # parser.add_argument("--drop_prob", type=float, default=0.5)

        return parent_parser


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:   
        super().__init__()
        self.h = heads
        self.d = d

        self.Wq = nn.Linear(d, d * heads, bias=False)
        self.Wk = nn.Linear(d, d * heads, bias=False)
        self.Wv = nn.Linear(d, d * heads, bias=False)

        # Unifying outputs at the end.
        self.unifyheads = nn.Linear(heads * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d]
        Returns:
            Self attention tensor of shape [b, l, d].
        """
        # b = batch
        # l = sequence length
        # d = embedding dim.

        b, l, d = x.size()
        h = self.h

        # Transform input embeddings of shape [b, l, d] to queries, keys and values.
        # The output shape is [b, l, d*h] which we transform into [b, l, h, d]. Then,
        # we fold the heads into the batch dimension to arrive at [b*h, l, d].
        queries = self.Wq(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        keys = self.Wk(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        values = self.Wv(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)

        # Compute raw weights of shape [b*h, l, l]
        w_prime = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)

        # Compute normalized weights by normalizing last dim. 
        # Shape: [b*h, l, l]
        w = F.softmax(w_prime, dim=-1)

        # Apply self attention to the values
        # Shape [b, h, l, d]
        out = torch.bmm(w, values).view(b, h, l, d)

        # Swap h, l back
        out = out.transpose(1, 2).contiguous().view(b, l, h * d)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    """
    A Transformer block consisting of self attention and ff-layers.
    Args:
        d (int): The embedding dimension
        heads (int): The number of attention heads
        n_mlp (int): n_mlp * d = hidden dim of independent FFNs
    """
    def __init__(self, d: int, heads: int, n_mlp: int) -> None:
        super().__init__()

        # The self-attention layer
        self.attention = SelfAttention(d=d, heads=heads)

        # The two layer norms
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # The feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(d, d * n_mlp),
            nn.ReLU(),
            nn.Linear(d * n_mlp, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input sequence embedding of shape [b, l, d]
        Returns:
            Transformer output tensor of shape [b, l, d]
        """
        # If there is no embedding -> d = 1. Have to unsqueeze it in last dim here.
        out = self.attention(x)
        out = self.norm1(out + x)
        out = self.ff(out) + out
        out = self.norm2(out)

        return out