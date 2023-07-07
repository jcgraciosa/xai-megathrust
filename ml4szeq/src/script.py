#!/usr/bin/env python
# coding: utf-8

# In[2]:


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import re
import sys
import random
import numpy as np
from functools import partial
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from matplotlib.lines import Line2D # for the legend

import torch
import yaml
from torch.utils.data import DataLoader

# Terrible hack to make sure Jupyter notebooks (which use different PYTHONPATH
# for some reason!) actually sees src/ directory so we can import from there.
os.chdir("/Users/jgra0019/Documents/codes/ml4szeq/ml4szeq")
#os.chdir("/home/jgra0019/xa94/jcg/ml4szeq/ml4szeq") # use this path in M3
print(f"--- Current working directory: {os.getcwd()}")
if not any([re.search("src$", path) for path in sys.path]):
    sys.path.append(str(Path.cwd() / "src"))

import default
from dataset import DFDataset
from fit import Fit
from utils import (convert_hyperparam_config_to_values, get_config,
                   get_full_hyperparam_config, load_data)

# %% GETTING HYPER-PARAMETERS
config_override_file = get_config("PARAMETER_YAML_FILE", None)
hyperparam_config = get_full_hyperparam_config(config_override_file=config_override_file)
print(
    "--- Hyperparamepters/metadata set as follows (may be altered later if using EXISTING wandb sweep):"
)
pprint(hyperparam_config)
params, _ = convert_hyperparam_config_to_values(hyperparam_config)


# In[3]:


num_train = 1
random.seed(43) # set the random seed 


# In[4]:


# %% PREPROCESSING - OR LOADING ALREADY PREPROCESSED - DATA
data_suffix = params.get("dataset", "16k")  # which dataset you want as input
data_folder = default.ROOT_DATA_DIRECTORY / data_suffix
use_cache = get_config("USE_CACHED_DATAFRAME", True)
use_cache = False
print(f"Preprocessing data ({use_cache=})...")
# Just getting the "params" for convenience

out_dict = {}
for i in range(num_train):

    print("Model: ", i)
    rand_seed = random.randint(0, 999999999) # generate a random seed to be used in sampling with replacement

    preprocessor = load_data(
        data_folder=data_folder,
        exclude_file=params["exclude_file"],
        target=params["target"],
        cats=params["mw_cats"],
        rand_seed = None, # for sampling with replacement 
        kernel_size=params["kernel_size"],
        use_cache=use_cache,
        protect_great=params["protect_great"]
    )
    print("Finished preprocessing! Number of features: ", len(preprocessor.inputs))
    
    #%% WANDB CONFIG
    # Determine if we want to use wandb to do sweeps/log this run
    use_wandb = get_config("USE_WANDB", None)

    # Define arguments to be passed into our training loop function.
    full_train_kwargs = dict(
        df=preprocessor.dataframe,
        inputs=preprocessor.inputs,
        model_name_add="test",
        hyperparam_config=hyperparam_config,
        use_wandb=use_wandb,
    )


    #%%
    # fit = Fit(**full_train_kwargs, fit_on_init=False)
    # df = preprocessor.dataframe
    # ds = fit.ds
    # dl = DataLoader(ds, sampler=df.index)
    # val = fit.validation_loop(dl)

    rand_seed = random.randint(0, 999999999)
    torch.manual_seed(rand_seed)
    if use_wandb:
        print("Using Weights and Biases!")
        import wandb
        wandb.login() # is this really needed? 

        # Get sweep id from config, otherwise create new sweep with specified params
        sweep_id = get_config("WANDB_SWEEP_ID", None)
        sweep_id = (
            sweep_id
            if sweep_id
            else wandb.sweep(
                hyperparam_config, project="ml4szeq", entity="jcgraciosa"
            )
        )
        wandb_func = partial(Fit, **full_train_kwargs)
        wandb.agent(
            sweep_id=sweep_id,
            function=wandb_func,
            project="ml4szeq",
            entity="jcgraciosa",
            count = 120
        )
    else:
        print("Not using Weights and Biases.")
        # full_train(**full_train_kwargs)
        fit_obj = Fit(**full_train_kwargs)
        out_dict[i] = fit_obj.out_df

