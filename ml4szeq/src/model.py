"""
All model related stuff is here.
- FCN --> Fully connected layer
- multiFCN --> multiple input features 
"""
from typing import Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class FC(nn.Module):
    def __init__(self, n_inp, n_out, do=None, bn=True, activ="relu"):  # activ
        super().__init__()
        self.bn = nn.BatchNorm1d(n_inp) if bn else None
        self.fc = nn.Linear(n_inp, n_out, bias=True)
        self.do = nn.Dropout(do) if do else None
        if activ == "relu":
            self.activ = nn.ReLU()
        elif activ == "leaky_1":
            self.activ = nn.LeakyReLU(0.1)
        elif activ == "leaky_2":
            self.activ = nn.LeakyReLU(0.2)
        elif activ == "prelu":
            self.activ = nn.PReLU()
        elif activ == "gelu":
            self.activ = nn.GELU()
        elif activ == "silu":
            self.activ = nn.SiLU()
        else:
            self.activ = Identity()

    def forward(self, x):
        x = self.bn(x) if self.bn else x
        x = self.fc(x)
        x = self.do(x) if self.do else x
        x = self.activ(x)
        return x


class multi_FCN(nn.Module):
    def __init__(
        self, in_feats, out_feats, sep_hidden, comb_hidden, num_inputs, do=0.1
    ):
        super().__init__()
        self.inputs = nn.ModuleList(
            [nn.Linear(in_feats, sep_hidden) for i in range(num_inputs)]
        )
        self.combine = nn.Linear(sep_hidden * num_inputs, comb_hidden[0], bias=True)

        self.layers = nn.Sequential(
            *[
                FC(comb_hidden[i], comb_hidden[i + 1], do=do)
                for i in range(len(comb_hidden) - 1)
            ]
        )

        self.out = nn.Linear(comb_hidden[-1], out_feats)

    def forward(self, xs):
        xs = [self.inputs[i](x) for i, x in enumerate(xs)]
        comb = torch.cat(xs, axis=1)
        comb = self.combine(comb)

        comb = self.layers(comb)
        # added relu activation function
        comb = torch.nn.functional.relu(comb)
        comb = self.out(comb)

        return comb


class EarthquakeMTL(nn.Module):
    """
    Class representing model to be used when performing classification and/or
    regression fits to earthquake data.
    """
    def __init__(
        self,
        n_cont,
        n_regions,
        n_out,
        hidden,
        emb=5,
        do=0,
        bn=True,
        activ="relu",
        categorical_output=True,
        regression_output=False
    ):
        super().__init__()

        self.num_emb = emb 
        # Regional embedding
        if self.num_emb > 0:
            self.emb = nn.Embedding(n_regions, emb)

        # Fully connected layers
        if emb > 0:
            self.inp = FC(n_cont + emb, hidden[0], do=do, bn=bn, activ=activ)
        else:
            self.inp = FC(n_cont, hidden[0], do=do, bn=bn, activ=activ)
        self.layers = nn.Sequential(
            *[
                FC(hidden[i], hidden[i + 1], do=do, bn=bn, activ=activ)
                for i in range(len(hidden) - 1)
            ]
        )

        # Switch between output modes
        self.categorical_output = categorical_output
        self.regression_output = regression_output
        
        if self.categorical_output:
            self.out_cat = nn.Linear(hidden[-1], n_out)
            
        if self.regression_output:
            self.out_reg = nn.Linear(hidden[-1], 1)
            self.sig_range = SigmoidRange(0, 1)  # Previously hard-coded to MW / 10

    def forward(self, x_cont) -> Tuple[Tensor, Tensor]:
        """
        Perform forward pass of the model on the input data `x_cont` and 
        `x_region`.

        Note that this always returns a tuple of (categorical output, regression
        output), but if the model was not initialised to do one of these outputs
        will return a None in its place. Therefore, always unpack the return value
        of this method into two variables like so:
        >>> cat, regr = model.forward(...)
        """
        x_cont = x_cont.float()
        # if self.num_emb > 0:
        #     x = torch.cat([x_cont, self.emb(x_region)], 1)
        # else:
        #     x = torch.cat([x_cont], 1)
        x = torch.cat([x_cont], 1)
        x = self.inp(x)
        x = self.layers(x)

        # Change output according to mode
        if self.categorical_output:
            out = self.out_cat(x) 
        if self.regression_output:
            out = self.out_reg(x)
        return out # set to regre since we will do some regression
