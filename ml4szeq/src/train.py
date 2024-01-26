import torch
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.std import tqdm

from model import EarthquakeMTL
from utils import ZERO_TENSOR


def train_loop(
    model: EarthquakeMTL,
    train_loader: DataLoader,
    cat_loss_fn: _Loss,
    regr_loss_fn: _Loss,
    optimizer: Optimizer,
    device = "cpu",
    scheduler: _LRScheduler = None,
    cat_scaling: float = 1,
    regr_scaling: float = 1,
):
    """
    Basic training function for an epoch, as given by `train_loader`.
    Will update model accordingly, and also step the given optimizer and learning
    rate scheduler. Returns the average training loss across the epoch.

    Parameters
    ----------
    model: 
    train_loader:
    cat_loss_fn: a categorical loss function
    regr_loss_fn: a regression loss function
    optimizer:
    device: e.g. cuda or "cpu". Defaults to cpu.
    scheduler: a learning rate scheduler. Defaults to None for no scheduling.
    cat_scaling: how much weight to give categorical loss
    regr_scaling: how much weight to give regression loss
    """
    model.train()
    
    total_cat_loss = 0
    total_regr_loss = 0
    total_comb_loss = 0

    # Loop through training data
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch, ((x_cont, x_region), (cat_labels, cont_labels)) in enumerate(pbar):
        # Move tensors to device (e.g. gpu) for (possibly) better performance
        x_cont, x_region, cat_labels, cont_labels = (
            x_cont.to(device),
            x_region.to(device),
            cat_labels.to(device),
            cont_labels.to(device),
        )

        # Get outputs and calculate loss.
        #cat, regr = model(x_cont, x_region)
        cat = model(x_cont) # for regression and classification
        
        
        # Calculate losses (ignoring cat/regr outputs if model doesn't have them),
        # and combining losses accordings to scaling factors.
        if model.categorical_output:
            cat_loss = cat_loss_fn(torch.squeeze(cat), torch.squeeze(cat_labels)) 
        if model.regression_output:
            cat_loss = regr_loss_fn(torch.squeeze(cat), torch.squeeze(cont_labels))
        
        comb_loss = cat_loss # *cat_scaling + regr_scaling * regr_loss

        # Add to cumulative losses so far
        total_cat_loss += cat_loss.item()
        #total_regr_loss += regr_loss.item()
        total_comb_loss += comb_loss.item()

        # Backpropagation 
        optimizer.zero_grad()
        comb_loss.backward() # Note we're using *combined* loss to backprop
        optimizer.step()

    # Step scheduler if it was provided
    if scheduler:
        scheduler.step()

    # Return dict containing metrics that might be of interest
    # NOTE: We divide by number of data points, NOT number of batches, since
    # last batch may not be a full batch, so its loss would skew the batch-average
    # down, which isn't so useful.
    #size = len(train_loader.dataset)
    # size = len(train_loader.sampler.indices)  # using SubsetRandomSampler
    size = train_loader.sampler.num_samples     # using WeightedRandomSampler 
   
    return {
        "cat_loss": total_cat_loss / size,
        #"regr_loss": total_regr_loss / size,
        "comb_loss": total_comb_loss / size
    }
