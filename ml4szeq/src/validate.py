from pickle import NONE
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EarthquakeMTL
from utils import ZERO_TENSOR


@torch.no_grad()
def validation_loop(
    model: EarthquakeMTL,
    val_loader: DataLoader,
    cat_loss_fn: _Loss,
    regr_loss_fn: _Loss,
    device="cpu",
    cat_scaling: float = 1,
    regr_scaling: float = 1,
):
    """
    Simple validation function. Runs model on input data loaded from `val_loader`,
    determines the categorical and regression loss (if applicable), and the
    accuracy for each target class, as well as the total accuracy overall (
    remember that total accuracy will depend on class balance in the data!).
    """
    model.eval()

    # Initialise metrics
    total_regr_loss = 0
    total_cat_loss = 0
    total_comb_loss = 0

    # Use these to keep track of predictions and labels across whole validation dataset
    #n_data = len(val_loader.dataset) # inflating number of class 0
    #n_data = len(val_loader.sampler.indices)   # using SubsetRandomSampler
    n_data = val_loader.sampler.num_samples     # using WeightedRandomSampler 
    all_predicted = np.zeros(n_data)
    all_labels = np.zeros(n_data)

    batch_size = val_loader.batch_size

    # Loop through validation data
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for batch, ((x_cont, x_region), (cat_labels, cont_labels)) in enumerate(pbar):
        # Move tensors to device (e.g. gpu) for (possibly) better performance
        x_cont, x_region, cat_labels, cont_labels = (
            x_cont.to(device),
            x_region.to(device),
            cat_labels.to(device),
            cont_labels.to(device),
        )
        #print(batch, cat_labels)
        # Get outputs and calculate loss.
        #cat, regr = model(x_cont, x_region)
        cat = model(x_cont)

        # Calculate losses (ignoring cat/regr outputs if model doesn't have them),
        # and combining losses according to scaling factors.
        if model.categorical_output:
            cat_loss = cat_loss_fn(torch.squeeze(cat), torch.squeeze(cat_labels)) 
        if model.regression_output:
            cat_loss = regr_loss_fn(torch.squeeze(cat), torch.squeeze(cont_labels)) 
        comb_loss = cat_scaling * cat_loss #+ regr_scaling * regr_loss

        # Add to cumulative losses
        total_cat_loss += cat_loss.item()
        #total_regr_loss += regr_loss.item()
        total_comb_loss += comb_loss.item()

        # Calculate accuracies
        if model.categorical_output:
            _, predicted = torch.max(cat, 1)
            _, cat_labels = torch.max(cat_labels, 1)
        
        if model.regression_output:
    
            predicted = torch.squeeze(cat)
            cat_labels = torch.squeeze(cont_labels)


        # Append labels and predicted values to relevant arrays in correct spots
        index = batch * batch_size
        all_predicted[index : index + len(predicted)] = predicted
        all_labels[index : index + len(predicted)] = cat_labels 

    if model.categorical_output:
        cr = classification_report(
            y_true=all_labels, y_pred=all_predicted, digits=3, output_dict=True, zero_division = 0
        )
        cm = confusion_matrix(y_true=all_labels, y_pred=all_predicted)
        # accuracy is a fine metric for *balanced* datasets
        accuracy = accuracy_score(y_true=all_labels,y_pred=all_predicted)
        f1 = cr["macro avg"]["f1-score"]
    if model.regression_output:
        cr = None
        cm = None
        accuracy = None
        f1 = None

    # If you want class-wise precision/recall that's in this dict, but I'm just
    # going to take the macro average F1. This seems to be the best for imbalanced
    # datasets, since it will equally penalise the model if it performs poorly
    # on a minority class as it would for a majority class. Check out the dict/
    # sckikit docs to see what other metrics you can take from here!
    

    # return loss value, accuracy
    out = {
        "cat_loss": total_cat_loss / n_data,
        #"regr_loss": total_regr_loss / n_data,
        "comb_loss": total_comb_loss / n_data,
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
    }
    #print(out)
    return out