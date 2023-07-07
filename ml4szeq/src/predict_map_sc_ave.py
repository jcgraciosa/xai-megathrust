#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import re
import sys
import pandas as pd
import numpy as np
from functools import partial
from pathlib import Path
from pprint import pprint
import copy
import seaborn as sns

import torch
import yaml
from torch.utils.data import DataLoader
from captum.attr import LRP 
from tqdm.std import tqdm

# for the LRP
import torch.nn as nn
from captum.attr import InputXGradient, LRP, IntegratedGradients
from captum.attr._utils.lrp_rules import (
    Alpha1_Beta0_Rule,
    EpsilonRule,
    GammaRule,
    IdentityRule,
)

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import RobustScaler
import vis_pkg
import helper_pkg
import json
import cartopy as cartopy
import cartopy.crs as ccrs

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cmocean
plt.rcParams["font.family"] = "Arial"
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3
mpl.rcParams.update({'hatch.color': 'gray'})

# Terrible hack to make sure Jupyter notebooks (which use different PYTHONPATH
# for some reason!) actually sees src/ directory so we can import from there.
os.chdir("/Users/jgra0019/Documents/codes/ml4szeq/ml4szeq")
print(f"--- Current working directory: {os.getcwd()}")
if not any([re.search("src$", path) for path in sys.path]):
    sys.path.append(str(Path.cwd() / "src"))

import default
from dataset import DFDataset
from fit import Fit
from utils import (convert_hyperparam_config_to_values, get_config,
                   get_full_hyperparam_config, load_data)
from model import *
from predictor import predictions

# %% GETTING HYPER-PARAMETERS
config_override_file = get_config("PARAMETER_YAML_FILE", None)
hyperparam_config = get_full_hyperparam_config(config_override_file=config_override_file)
print(
    "--- Hyperparameters/metadata set as follows (may be altered later if using EXISTING wandb sweep):"
)
pprint(hyperparam_config)


# # Load model/s, predict, perform XAI, get average
# 

# In[2]:


''' settings to define '''

# scenario 4 - final models
scenario = 4 # scenario 3 - ~200 params; scenario 4 - ~180 params
do_stnd = True # perform standardization of the relevance values or not
with_large = True # set to True if working with regions with large earthquakes
apply_thresh = False # set to True if we apply threshold to heatmap values

# for thresholds
at_least = 4 # number of regions FINAL - 05/07/2023
thresh_val = 0.1
    
hparam_file = Path("/Users/jgra0019/Documents/codes/ml4szeq/ml4szeq/parameters/no_rand_ctrl_val_0.30.json")

if scenario == 3:
    if with_large:
        region_list = ["sam", "sum", "alu", "kur"]
        model_list = [  "800_200_16_1e-01",
                        "400_800_8_1e-02",
                        "200_800_16_1e-01",
                        "400_400_16_1e-02"]
        epoch_list = [2, 7, 5, 7]
   
    else:
        region_list = ["ryu", "cam", "izu", "ker"]
        model_list = ["200_200_32_1e-02"]
        epoch_list = [21]
elif scenario == 4: # this is the final version to use!!!

    with open(hparam_file) as json_data_file:
        hparam = json.load(json_data_file)

    hparam_sset_v2 = hparam["v2"]
    
    if with_large:
        region_list = ["sam", "sum", "alu", "kur"]
        model_list = [  1,      # ranks for each in the region
                        7,      # one-to-one correspondence
                        1,
                        2]
        epoch_list = None
   
    else:
        pass
        #region_list = ["ryu", "cam", "izu", "ker"]
        #model_list = ["200_200_32_1e-02"]
        #epoch_list = [21]
        # not implemented 

# for the model_list, list the part with hyperparameter list
num_class = 3
num_model = 1
device = "cpu"

# xai parameters 
algo_use = "lrp_def" # use default

if scenario == 3:
    scen_dir = "scenario3" # for models
    if do_stnd:
        out_folder = "scenario3" 
    else:
        out_folder = "scenario3-no-stnd"

elif scenario == 4: 
    scen_dir = "scenario3" #for models # this is the final - May 06
    
    if do_stnd:
        out_folder = "scenario3"
    else:
        out_folder = "scenario3-no-stnd"


og_model_dir = Path("/Users/jgra0019/Documents/codes/ml4szeq/ml4szeq/out/models") # exclude region
out_dir = Path("/Users/jgra0019/Documents/codes/ml_proj1/ml_proj1/out")

og_model_dir = og_model_dir/scen_dir
out_dir = out_dir/out_folder

# if region in ["ryu", "cam", "izu", "ker"]:
#     model_dir = model_dir/"ryu"
# else:
#     model_dir = model_dir/region


# In[3]:


for iter, region in enumerate(region_list):
    print(region)


# ### A. Preparations for the map and function declarations

# In[4]:


''' Function declarations '''

# arrange features properly
def arrange_features(feat_list):
    phys_state_list = ["CRD_UP", "CRS_UP", "CRM_UP",
                   "INV_UP", "DLT_UP", "SED", "SRO", "IRO", "LRO"
                   ]
    dyna_list = ["FRE_DG", "FRE_UP", "BGR_DG",  
                "EGO_UP", 
                "EGO_L_UP", 
                "EGO_SL_UP",  
                "EGO_UM_UP",
                "EGR_DG", 
                "EGR_UP", 
                "EGR_BG_UP",
                "DXT", "FDM", "SDM"]
    kine_list = ["V_UP", "V_TN", "AGE"]
    
    new_feat_list = []

    # loop through all phys state list
    for phys_state in phys_state_list:
        for feat in feat_list:
            if (phys_state in feat) and (feat not in new_feat_list):
                new_feat_list.append(feat)
            
    # loop through all dyna list
    for dyna in dyna_list:
        for feat in feat_list:
            if (dyna in feat) and (feat not in new_feat_list):
                new_feat_list.append(feat)
    
    # loop through all kinematic state list
    for kine in kine_list:
        for feat in feat_list:
            if (kine in feat) and (feat not in new_feat_list):
                new_feat_list.append(feat)

    # if scenario == 3:
    #     new_feat_list = new_feat_list + ["TRG_STD1", "TRG_STD2", "TRG_STD3", "TRG_STD4", "TRG_STD5", "RND_CTRL"]
    # else:
    #     new_feat_list.append("RND_CTRL")

    return new_feat_list


# In[5]:


'''Preparations for XAI plot'''
xtick_lab = ["Curvature (along-dip)", 
            "Curvature (along-strike)", 
            "Curvature (mean)",
            "2nd strain inv. (UP)", 
            "Dilatation (UP)",   
            "Sediment thickness", 
            r"Roughness (short $\lambda$)",
            r"Roughness (intermediate $\lambda$)", 
            r"Roughness (long $\lambda$)",  
            "Free air gravity anomaly (DG)",
            "Free air gravity anomaly (UP)",
            #"Bouguer gravity anomaly (UP)", # is actually DG
            "EGM 2008 geoid (UP)",
            "EGM 2008 geoid L (UP)",
            "EGM 2008 geoid SL (UP)",
            "EGM 2008 geoid UM (UP)",
            "EGM 2008 Free air gravity anomaly (DG)",
            "EGM 2008 Free air gravity anomaly (UP)",
            "EGM 2008 Bouguer anomaly (UP)",
            "Slab depth", 
            "Depth grad. (magnitude)",
            "Depth curv. (magnitude)",
             "Plate motion"]
            #"Null model"]

# if scenario == 3:
#     #borders = [18, 42, 48, 66, 69, 72, 84, 96, 108, 120,
#     #        132, 144, 156, 168, 180, 192, 204, 216, 228]
#     borders = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 69, 72, 75, 78, 81, 84, 
#               87, 90, 93, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156,
#               162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228]
# elif scenario == 4:
#     borders = [15, 35, 40, 55, 58, 60, 68, 76, 86, 96,
#                 106, 116, 126, 136, 146, 156, 166, 176, 186] # to edit
borders = [ 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 25, 
            27, 29, 31, 33, 35, 37, 40, 43, 46]
# create the colormap - use different colormap for average
cmap_use = cmocean.cm.tempo
cmap = cmocean.tools.crop_by_percent(cmap_use, per = 10, which='max', N=None)
# hmap_labels = [ r"$M_w < 6.5$",
#                 r"$6.5 \leq M_w < 8.4$", 
#                 r"$M_w \geq 8.4$"
#             ]

hmap_labels = [ 
                r"average $\tilde{R}$", 
                r"average $\tilde{R}$", 
                r"average $\tilde{R}$"   
            ]

# hmap_labels = [ 
#                 r"$\tilde{R} \geq$ " + "{:.2f}".format(thresh_val), 
#                 r"$\tilde{R} \geq$ " + "{:.2f}".format(thresh_val), 
#                 r"$\tilde{R} \geq$ " + "{:.2f}".format(thresh_val)    
#             ]

if do_stnd:
    bin_use = np.linspace(-3, 3)
else:
    bin_use = np.linspace(-5, 5)


# ### B. Loop through all the models, perform predictions, and map

# In[13]:


from tkinter import N

ave_relevance = None

for iter, region in enumerate(region_list):

    if scenario == 3: # old
        if with_large:
            model_dir = model_dir/region
            model_name = model_list[iter]
            epoch = epoch_list[iter]
        else:
            model_dir = og_model_dir/"ryu"
            model_name = model_list[0] # first element
            epoch = epoch_list[0] # first element

        for subdir, dirs, files in os.walk(model_dir): # get the list of models inside model_dir
            break
    
        for dir in dirs:
            if model_name in dir:
                use_model = dir
                
        print(use_model)

        model_params = model_name.split("_")

        ''' MACHINE LEARNING STUFF HERE '''

        # set-up hyperparameters - override values in default.yml 
        hyperparam_config["parameters"]["exclude_file"]["value"] = region + ".csv" 
        hyperparam_config["parameters"]["hidden_layers"]["value"] = [int(model_params[0]), int(model_params[1])]
        hyperparam_config["parameters"]["batch_size"]["value"] = int(model_params[2])
        hyperparam_config["parameters"]["learning_rate"]["value"] = float(model_params[3])
        
        epoch_use = epoch
        epoch_use = epoch_use # best epoch in title of plot is already - 1
    elif scenario == 4: # final version
        # get the hyperparameter sset 
        hparam_sset = hparam_sset_v2[region][str(model_list[iter])]

        epoch_use = hparam_sset["epoch_use"]
        use_model = hparam_sset["folder"]
        use_model = use_model.replace("/", ":")

         # set-up hyperparameters - override values in default.yml 
        hyperparam_config["parameters"]["exclude_file"]["value"] = region + ".csv" 
        hyperparam_config["parameters"]["hidden_layers"]["value"] = hparam_sset['hidden_layers']
        hyperparam_config["parameters"]["batch_size"]["value"] = hparam_sset["batch_sz"]
        hyperparam_config["parameters"]["learning_rate"]["value"] = float(hparam_sset["lr"])
        
    ##################
    params, _ = convert_hyperparam_config_to_values(hyperparam_config) # convert here to include whatever were overriden
    data_suffix = params.get("dataset", "16k")  # which dataset you want as input
    data_folder = default.ROOT_DATA_DIRECTORY / data_suffix
    use_cache = False
   
    preprocessor = load_data(
            data_folder=data_folder,
            exclude_file=params["exclude_file"],
            target=params["target"],
            cats=params["mw_cats"],
            rand_seed = None, # for sampling with replacement
            kernel_size=params["kernel_size"],
            skip_drop_na = True,
            rd_exclude = True,
            use_cache=use_cache,
            protect_great=params["protect_great"]
        )
    
     # Define arguments to be passed into our testing loop function.
    full_pred_kwargs = dict(
        df=preprocessor.dataframe,
        inputs=preprocessor.inputs,
        hyperparam_config=hyperparam_config,
        model_name_add=None,
        use_wandb=False,
    )

    # create pred_ds 
    pred_ds = DFDataset(dataframe = preprocessor.dataframe, 
                        inputs=preprocessor.inputs, 
                        target=preprocessor.target, 
                        force_cats = 5)
    pred_dl = torch.utils.data.DataLoader(pred_ds, batch_size = 1, shuffle=False)

    ################## PERFORM PREDICTION 
    model_fname = "epoch-" + str(epoch_use) + ".pt"
    idx = 0

    ''' SECTION ON XAI '''
    model_path = og_model_dir/region/use_model/model_fname # read the model again
    pred_obj = Fit(fit_on_init = False, **full_pred_kwargs, force_cats = 0) # initialize lang pirm
    pred_model = copy.deepcopy(pred_obj.model)
    pred_model.load_state_dict(torch.load(model_path))
    pred_model.to(device)
    pred_model.eval()

    # set the rule here if needed, then proceed with creation of LRP object
    if algo_use == "lrp_def":   
        print("Default")
    elif algo_use == "lrp_epsilon":
        pred_model.inp.rule = EpsilonRule(1e-5)          
        pred_model.layers.rule = EpsilonRule(1e-5)   
        pred_model.out_cat.rule = EpsilonRule(1e-5)     
    elif algo_use == "lrp_alpha1_beta0":
        pred_model.inp.rule = Alpha1_Beta0_Rule()           
        pred_model.layers.rule = Alpha1_Beta0_Rule()
        pred_model.out_cat.rule = Alpha1_Beta0_Rule()   
    elif algo_use == "lrp_gamma":
        pred_model.inp.rule = GammaRule()           
        pred_model.layers.rule = GammaRule()
        pred_model.out_cat.rule = GammaRule()
    elif algo_use == "int_grad":
        None

    # for int_grad 
    if algo_use != "int_grad":
        attr_obj = LRP(pred_model)
    else: # integrated gradient
        attr_obj = IntegratedGradients(pred_model)

    # create the containers of the attribution value and convergence delta
    c0_attr = None
    c1_attr = None
    c2_attr = None

    c0_delta = None
    c1_delta = None
    c2_delta = None

    # start of additional
    # use data loaders given above
    for i, ((x_cont, x_region), (cat_labels, cont_labels)) in enumerate(pred_dl): 
        x_cont, x_region, cat_labels, cont_labels = (
            x_cont.to(device),
            x_region.to(device),
            cat_labels.to(device),
            cont_labels.to(device),
        )

        # Get outputs and calculate loss
        cat = pred_model(x_cont)
        pred_vals = torch.sigmoid(cat)
        _, pred_vals2 = torch.max(torch.sigmoid(cat), 1)

        # set predicted value as the target, then find out why the model chose to fire up this node
        if algo_use != "int_grad":
            attribution, conv = attr_obj.attribute( x_cont, 
                                                target = pred_vals2.item(), # so why is the predicted value the target? 
                                                #target = torch.argmax(cat_labels).item(),  # think about the correct target
                                                return_convergence_delta = True)
        else: # integrated gradients
            attribution, conv = attr_obj.attribute( x_cont, 
                                                n_steps = 500,
                                                target = pred_vals2.item(),
                                                #target = torch.argmax(cat_labels).item(),  # think about the correct target
                                                return_convergence_delta = True)

        # standardization of attribution values # FIXME: note, that because of this conv will not mean anything
        test = attribution.cpu().detach().numpy()
        if do_stnd: # perform standardization of relevance values
            transformer = RobustScaler().fit(test.T)
            rescaled = transformer.transform(test.T)  # replace attribution with the scaled version
        else:
            rescaled = test.T

        add_cond = True

        # separate by class
        if (pred_vals2.item() == 0) and add_cond: # class 0
            if c0_attr is None:
                c0_attr = rescaled
                c0_delta = conv
            else:
                c0_attr = np.concatenate((c0_attr, rescaled), axis = 1)
                c0_delta = torch.cat((c0_delta, conv), axis = 0)

        elif (pred_vals2.item() == 1) and add_cond: # class 1
            if c1_attr is None:
                c1_attr = rescaled
                c1_delta = conv
            else:
                c1_attr = np.concatenate((c1_attr, rescaled), axis = 1)
                c1_delta = torch.cat((c1_delta, conv), axis = 0)
        
        elif (pred_vals2.item() == 2) and add_cond: # class 2
            if c2_attr is None:
                c2_attr = rescaled
                c2_delta = conv
            else:
                c2_attr = np.concatenate((c2_attr, rescaled), axis = 1)
                c2_delta = torch.cat((c2_delta, conv), axis = 0)

    # then transpose 
    c0_attr = c0_attr.T if c0_attr is not None else None
    c1_attr = c1_attr.T if c1_attr is not None else None
    c2_attr = c2_attr.T if c2_attr is not None else None

    # # calculate the mean and standard deviation of the relevance 
    # try to use median
    c0_mean_out = np.median(c0_attr, axis = 0) if c0_attr is not None else None
    c1_mean_out = np.median(c1_attr, axis = 0) if c1_attr is not None else None
    c2_mean_out = np.median(c2_attr, axis = 0) if c2_attr is not None else None

    c0_std_out = c0_attr.std(axis = 0) if c0_attr is not None else None
    c1_std_out = c1_attr.std(axis = 0) if c1_attr is not None else None
    c2_std_out = c2_attr.std(axis = 0) if c2_attr is not None else None

    # convert tensor to numpy array
    c0_delta    = c0_delta.detach().numpy() if c0_delta is not None else None
    c1_delta    = c1_delta.detach().numpy() if c1_delta is not None else None
    c2_delta    = c2_delta.detach().numpy() if c2_delta is not None else None

    # create the necessary dataframes in here
    # one for the mean and standard deviation
    # another one for the relevance values for each feature and sample
    c0_mean_rel_df = {"FEATURE" : pred_dl.dataset.inputs,
                    "AVE_REL": c0_mean_out,
                    "STD_REL": c0_std_out}
    c1_mean_rel_df = {"FEATURE" : pred_dl.dataset.inputs,
                    "AVE_REL": c1_mean_out,
                    "STD_REL": c1_std_out}
    c2_mean_rel_df = {"FEATURE" : pred_dl.dataset.inputs,
                    "AVE_REL": c2_mean_out,
                    "STD_REL": c2_std_out}

    # dataframes where feature name is the column, then the relevance values are the rows
    c0_rel_df = {}
    c1_rel_df = {}
    c2_rel_df = {}
    for i, feat in enumerate(pred_dl.dataset.inputs): # feature name as column name, then values as rows
        c0_rel_df[feat] = c0_attr[:, i] if c0_attr is not None else None
        c1_rel_df[feat] = c1_attr[:, i] if c1_attr is not None else None
        c2_rel_df[feat] = c2_attr[:, i] if c2_attr is not None else None

    # convert to dataframes
    c0_mean_rel_df = pd.DataFrame.from_dict(c0_mean_rel_df) #sort_values(ascending = False, by = "AVE_REL").reset_index(drop = True)
    c1_mean_rel_df = pd.DataFrame.from_dict(c1_mean_rel_df) #sort_values(ascending = False, by = "AVE_REL").reset_index(drop = True)
    c2_mean_rel_df = pd.DataFrame.from_dict(c2_mean_rel_df) #sort_values(ascending = False, by = "AVE_REL").reset_index(drop = True)

    c0_rel_df = pd.DataFrame.from_dict(c0_rel_df if c0_attr is not None else [c0_rel_df])
    c1_rel_df = pd.DataFrame.from_dict(c1_rel_df if c1_attr is not None else [c1_rel_df])
    c2_rel_df = pd.DataFrame.from_dict(c2_rel_df if c2_attr is not None else [c2_rel_df])

    # the dataframes that we need - only up to 3 classes
    to_concat = [   c0_rel_df.median(axis = 0).to_frame().transpose(), 
                    c1_rel_df.median(axis = 0).to_frame().transpose(), 
                    c2_rel_df.median(axis = 0).to_frame().transpose()]
    all_median_df = pd.concat(to_concat).reset_index(drop = True)

    sorted_feat = arrange_features(all_median_df.columns)

    all_median_df = all_median_df[sorted_feat]
    
    # plot the XAI results
    all_median_df = all_median_df.transpose() # transpose
    all_median_df = all_median_df.iloc[:, ::-1] # reverse order of columns

    all_median_df[all_median_df < 0] = 0

    if apply_thresh:
        # threshold using quantile values
        # thresh = all_median_df.quantile(0.8)
        # # thresh[0] # threshold for 2
        # # thresh[1] # threshold for 1
        # #thresh[2] # threshold for 0

        # all_median_df.loc[all_median_df[2] < thresh[0], 2] = 0 # class 2 - thresh 0
        # all_median_df.loc[all_median_df[1] < thresh[1], 1] = 0 # class 1 - thresh 1
        # all_median_df.loc[all_median_df[0] < thresh[2], 0] = 0 # class 0 - thresh 2

        # hard set threshold value
        all_median_df.loc[all_median_df[2] < thresh_val, 2] = 0 # class 2 - thresh 0
        all_median_df.loc[all_median_df[1] < thresh_val, 1] = 0 # class 1 - thresh 1
        all_median_df.loc[all_median_df[0] < thresh_val, 0] = 0

        all_median_df.loc[all_median_df[2] >= thresh_val, 2] = 1 # class 2 - thresh 0
        all_median_df.loc[all_median_df[1] >= thresh_val, 1] = 1 # class 1 - thresh 1
        all_median_df.loc[all_median_df[0] >= thresh_val, 0] = 1

    if ave_relevance is None:
        ave_relevance = copy.deepcopy(all_median_df)
    else:
        ave_relevance = ave_relevance.add(all_median_df, fill_value = 0) 

# calculate the mean
ave_relevance = ave_relevance/at_least
# ave_relevance.loc[ave_relevance[2] < at_least, 2] = 0
# ave_relevance.loc[ave_relevance[2] >= at_least, 2] = 1
# ave_relevance.loc[ave_relevance[1] < at_least, 1] = 0
# ave_relevance.loc[ave_relevance[1] >= at_least, 1] = 1
# ave_relevance.loc[ave_relevance[0] < at_least, 0] = 0
# ave_relevance.loc[ave_relevance[0] >= at_least, 0] = 1

## Plot the results of XAI analysis
for i in range(3):
    fig, ax = plt.subplots(dpi = 300, figsize = (0.7, 15))
    if do_stnd:
        sns.heatmap(ave_relevance[[i]], cmap = cmap, vmin = 0, vmax = 1.5, linewidths = 1e-3, linecolor = "gray" #cbar = False
                                , cbar_kws={'label': r"average $\tilde{R}$", 
                                "shrink" : 3,
                                "location": "bottom", 
                                "orientation": "horizontal",
                                "pad": 0.02,
                                "ticks": [0, 0.5, 1.0, 1.5]}
                    )
    else:
        sns.heatmap(ave_relevance[[i]], cmap = cmap, vmin = 0, vmax = 0.3, linewidths = 1e-3, linecolor = "gray" #cbar = False
                                , cbar_kws={'label': r"$<\tilde{R}>$", 
                                "shrink" : 3,
                                "location": "bottom", 
                                "orientation": "horizontal",
                                "pad": 0.02,
                                "ticks": [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]}
                    )
            

    # broken lines
    ax.hlines(borders, 0, 1, linestyle = "-", colors = "k", linewidth = 0.8)

    bnds = np.array([0] + borders + [borders[-1] + 3])
    bnds = 0.5*(bnds[1:] + bnds[:-1])

    if thresh_val == 1:
        ax.set_yticks(bnds)
        ax.set_yticklabels(xtick_lab, rotation = 0, fontsize = 10)
    else:
        ax.set_yticks([])

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels([hmap_labels[i]],
                    rotation = -30, ha = "right")

    # save the heatmap
    if with_large:
        if apply_thresh:
            plt_fname = out_dir/("large_eq_ave_thresh_" + str(thresh_val) + "_at_least_" + str(at_least) + "_hmap_c" + str(i) + ".png")
        else:
            plt_fname = out_dir/("large_eq_ave_hmap_c" + str(i) + ".png")
    else:
        if apply_thresh:
            plt_fname = out_dir/("no_large_eq_ave_thresh_" + str(thresh_val) + "_at_least_" + str(at_least) + "_hmap_c" + str(i) + ".png")
        else:
            plt_fname = out_dir/("no_large_eq_ave_hmap_c" + str(i) + ".png")
            
    plt.savefig(plt_fname, dpi = "figure", bbox_inches='tight')
    plt.close()


# In[12]:


ave_relevance[[2]].max()


# In[8]:


plt_fname


# In[9]:


sorted_feat


# In[ ]:




