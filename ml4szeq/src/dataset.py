"""
Dataset classes used to package dataframes for ease of access to columns, target variables and converting items into tensors
"""
from functools import cached_property
from typing import List, Tuple
from pandas.core.frame import DataFrame

import torch
from torch.functional import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset


class DFDataset(Dataset):
    """
    A simple wrapper class that converts pandas DataFrames into pytorch
    DataSets.
    """

    def __init__(self, dataframe: DataFrame, inputs: List[str], target: str, force_cats = 0):
        """
        Parameters
        ----------
        dataframe: pandas.DataFrame
            Preprocessed DataFrame containing all variables of interest.
        inputs: List[str]
            List of strings indicating which columns in DataFrame are to be used
            as inputs into model. Excludes "REGION", which is treated separately.
        target: str
            Target variable for regression. E.g. "MW".
        force_cats: set to more than 0 if you want to override the original behavior
        """
        self.dataframe = dataframe
        self.inputs = inputs
        self.targets = target
        self.force_cats = force_cats

    def __len__(self):
        """Length of DataFrame."""
        return len(self.dataframe)

    def __getitem__(self, i) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Returns row of data from *label-based* index `i` as tuple of:
            (
                (inputs, region),
                (categorical label, regression label)
            )
        NOTE: Emphasis on *label-based*. Pandas DataFrame have two types of
        indexing: label-based and positional. Label-based is my preference since
        shuffling a DataFrame will not mutate its labels, meaning a label-index
        is stable! Remembering that some rows get dropped during preprocessing,
        there may be no row with index 0, for example!
        """
        regression_label = self.dataframe.loc[i, self.targets]
        regression_label = torch.tensor(regression_label).float()
        #regression_label = torch.tensor(regression_label).float() / 10
        x, x_region = (
            self.dataframe.loc[i, self.inputs],
            self.dataframe.loc[i, "REGION"],
        )

        cat = self.dataframe.loc[i, "MW_CAT"]
        cat_label = self._make_label(cat)

        return (
            (torch.tensor(x).float(), torch.tensor(x_region).int()),
            (cat_label.float(), regression_label),
        )

    @cached_property
    def num_cats(self):
        """Number of target categories found in DataFrame."""
        return len(self.dataframe["MW_CAT"].unique())

    def _make_label(self, i, great=True):
        """
        Returns one-hot encoded vector to represent label for category of
        magnitude of earthquake.
        """
        # Default to one hot encoding
        if self.force_cats == 0:
            return F.one_hot(torch.tensor(i).long(), num_classes=self.num_cats)
        else:
            return F.one_hot(torch.tensor(i).long(), num_classes=self.force_cats)

