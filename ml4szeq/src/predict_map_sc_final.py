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
import random
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
                             confusion_matrix, f1_score)
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


# ## Final version of mapping code

# In[2]:


''' settings to define '''

region = "kur"
mod_rank = 2 # rank of the model - valid numbers are 1, 2, 3, 4, 5

scenario = 4 # 3 - with RAND_CTRL; 4 - without RAND_CTRL
do_stnd = True # perform standardization of the relevance values or not
use_tp = True # use True positive - will only work with sam
make_map = True # set to True if you want to produce the maps
apply_thresh = False # set to True if we apply threshold to heatmap values
do_train = False # set to True if we train the model from scratch

thresh_val = 0.5

hparam_file = Path("/Users/jgra0019/Documents/codes/ml4szeq/ml4szeq/parameters/no_rand_ctrl_val_0.30.json")

with open(hparam_file) as json_data_file:
    hparam = json.load(json_data_file)

# select scenario sset
if scenario == 3:
    hparam_sset = hparam["v1"]
elif scenario == 4:
    hparam_sset = hparam["v2"]

# get the hyperparameter sset 
hparam_sset = hparam_sset[region][str(mod_rank)]

if not do_train:
    epoch_use = hparam_sset["epoch_use"]
    folder_use = hparam_sset["folder"]
    folder_use = folder_use.replace("/", ":")

# for the model_list, list the part with hyperparameter list
num_class = 3
num_model = 1
device = "cpu"

# xai parameters 
algo_use = "lrp_def" # use default

if scenario == 3:
    scen_dir = "scenario3" # for models
    
    if do_stnd:
        if not use_tp:
            out_folder = "scenario3"    
        else:
            out_folder = "scenario3-tp"   
    else:
        out_folder = "scenario3-no-stnd"

elif scenario == 4:
    scen_dir = "scenario3" #for models
    
    if do_stnd:
        out_folder = "scenario4"
    else:
        out_folder = "scenario4-no-stnd"

model_dir = Path("/Users/jgra0019/Documents/codes/ml4szeq/ml4szeq/out/models") # exclude region
out_dir = Path("/Users/jgra0019/Documents/codes/ml_proj1/ml_proj1/out")

model_dir = model_dir/scen_dir
out_dir = out_dir/out_folder

if region in ["ryu", "cam", "izu", "ker"]:
    model_dir = model_dir/"ryu"
else:
    model_dir = model_dir/region
out_dir = out_dir/region # this is correct
print("output directory: ", out_dir)


# In[3]:


hparam_sset


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


''' 
preparations for the map of the predictions
run before the loop since these don't really change
'''

# earthquake ruptures to map
eq_file_dict = {"alu": ["alu04", "alu06", "alu07", "alu08", "alu05"],
                "kur": ["kur00", "kur13", "kur02", "kur20", "kur12", "kur17", 
                        "kur14", "kur01", "kur19", "kur24", "kur03"],
                "sam": ["sam29", "sam11", "sam18", "sam24", "sam15", "sam07", "sam10",
                        "sam22", "sam09", "sam08", "sam04", "sam03", "sam06", "sam12",
                        "sam17", "sam26", "sam02", "sam28", "sam05", "sam19"],
                "sum": ["sum05", "sum04", "sum08", "sum00", "sum02",
                        "sum03", "sum06"],
                "cam": ["cam03", "cam10", "cam07", "cam11", "cam01", "cam09",
                        "cam17", "cam12", "cam04", "cam00", "cam15"],
                "ker": ["ker02", "ker01", "ker00"],
                "ryu": ["ryu00", "ryu01", "ryu02"],
                "izu": []               
            }

conf_dir = Path("/Users/jgra0019/Documents/codes/region-paths")
conf_fname = conf_dir/(region + "_conf.json")
map_set_fname = "/Users/jgra0019/Documents/codes/globdat-paths/map_setting_conf.json"

with open(conf_fname) as json_data_file:
    conf = json.load(json_data_file)

with open(map_set_fname) as data_file:
    #print(map_set_fname)
    all_map_setting = json.load(data_file)

map_setting = all_map_setting[region]
root_dir = Path(conf["mac_root"])
other_settings = conf["conf"]
conf = conf["path"]

##########################
# Map settings
##########################

# convert range of longitude to -180 to 180 if alu
is_alu = False
if any(re.findall(r'alu', region, re.IGNORECASE)):
    is_alu = True
is_ker = False
if any(re.findall(r'ker', region, re.IGNORECASE)):
    is_ker = True

# read trench data
trench_plt = pd.read_csv(root_dir/conf["trench_full"], sep=',', header = None)
trench_plt.columns = ['LON', 'LAT']

# text labels 
label_fname = root_dir/conf["map_label"]
label_df = pd.read_csv(label_fname, header = 'infer')
label_df = label_df[label_df["TEXT"] != "Reference (0 km)"] # remove the reference - replace with a red marker
label_df = label_df[label_df["IS_EQ"] != True] # remove earthquake labels
label_df.loc[label_df["COLOR"] == 'yellow', "COLOR"] = 'firebrick'

# earthquake slip files
fdir = root_dir/conf["slip_line2"]

eq_files = eq_file_dict[region]

eq_outlines = {}
# iterate opening
cnt = 0
for subdir, dirs, files in os.walk(fdir):
    for file in files:
        file_id = file.replace("_sqk_", "").replace("_rup.csv", "")
        if file_id in eq_files:
            fname = fdir/file
            eq_df = pd.read_csv(fname, sep = ',', comment = "#")
            #print(eq_df)
            #print(fname)
            if eq_df.shape[1] > 2:
                eq_df = eq_df[['LON', 'LAT', 'COUNT']]
                eq_df = eq_df.sort_values(by = 'COUNT')

            eq_outlines[cnt] = eq_df
            cnt += 1


# other settings
arr_sz = map_setting["trench_arr_sz"] # trench arrow size
rel_pos = map_setting["label_rel_pos"] # for the label
contour_fsize = map_setting["contour_fnt_sz"]  # contour label font size
lwidth = map_setting["contour_lwidth"] # contour line width
label_x = map_setting["region_lab_x"] # region label x
label_y = map_setting["region_lab_y"] # region label y
as_line = map_setting["eq_as_line"]

# generate grid data since you need to do some revisions
''' Make sure these are correct!!! Especially ds value. '''
n_max = 300 # in km in direction of downgoing plate
n_min = -300 # in km in direction of upper plate
dn = 300 # step in the n-axis
ds = 50 # step in the s-axis 

# Open the topo grid file
# process topo grid file
topo_fname = root_dir/conf['topo_grd']
lon_topo, lat_topo, elev, topo_hs = vis_pkg.process_topo(topo_fname)

# since i don't want to write new code, make a work-around

feat_df = pd.read_csv(root_dir/conf["dilat"], header =None, sep = "\t")
feat_df = feat_df.dropna()
feat_df = feat_df.reset_index(drop=True)
feat_df.columns = ["LON", "LAT", "VAL"]
feat_df["LON"] = feat_df["LON"]%360 # make sure to convert 

n_ax, s_ax, lon_grid, lat_grid = helper_pkg.make_grid(root_dir/conf["grid_dir_" + str(ds)], 
                                                    n_max = n_max, n_min = n_min, 
                                                    dn = dn, ds = ds, 
                                                    ncdf_fname = None)
lon_grid = lon_grid%360



workaround_df = helper_pkg.map_data_to_grid(s_ax = s_ax, n_ax = n_ax, 
                                    lon_grid = lon_grid, lat_grid = lat_grid, 
                                    dep_df = None,  
                                    in_data_df = feat_df, 
                                    mode = 0, 
                                    rm_unmapped = True)


# setting for the sort_by - for the PatchCollection
if region in ["alu"]:
    sort_by = 0
elif region in ["sam", "kur", "sum"]:
    sort_by = 1
else:
    sort_by = 1

# colormaps
cmap = matplotlib.cm.get_cmap('Set3') # colormap used
color_list = [  matplotlib.colors.rgb2hex(cmap(6/12)), 
                matplotlib.colors.rgb2hex(cmap(5/12)), # 5/12 
                matplotlib.colors.rgb2hex(cmap(3/12)) ]


# In[6]:


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

borders = [ 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 25, 
            27, 29, 31, 33, 35, 37, 40, 43, 46]
            #51]

# 2 rows each, except for the non-grid data
# if scenario == 3:
#     borders = borders + [52, 53, 54, 55, 56]
#     xtick_lab = xtick_lab + ["1 std", "2 std", "3 std", "4 std", "5 std"]

# create the colormap
cmap_use = cmocean.cm.amp
cmap = cmocean.tools.crop_by_percent(cmap_use, per = 10, which='max', N=None)

hmap_labels = [ r"$M_w < 7.7$",
                r"$7.7 \leq M_w < 8.5$", 
                r"$M_w \geq 8.5$"
            ]
if do_stnd:
    bin_use = np.linspace(-3, 3)
else:
    bin_use = np.linspace(-5, 5)


# In[7]:


hyperparam_config


# ### B. Loop through all the models, perform predictions, and map

# In[8]:


''' MACHINE LEARNING STUFF HERE '''
random.seed(43) # set the random seed 

# set-up hyperparameters - override values in default.yml 
hyperparam_config["parameters"]["exclude_file"]["value"] = region + ".csv" 
hyperparam_config["parameters"]["dropout"]["value"] = float(hparam_sset["dropout"])
hyperparam_config["parameters"]["hidden_layers"]["value"] = hparam_sset['hidden_layers']
hyperparam_config["parameters"]["batch_size"]["value"] = hparam_sset["batch_sz"]
hyperparam_config["parameters"]["learning_rate"]["value"] = float(hparam_sset["lr"])

if do_train:
    params, _ = convert_hyperparam_config_to_values(hyperparam_config)

    data_suffix = params.get("dataset", "16k")  # which dataset you want as input
    data_folder = default.ROOT_DATA_DIRECTORY / data_suffix
    use_cache = get_config("USE_CACHED_DATAFRAME", True)
    use_cache = False

    # train the model first
    ##################
    preprocessor = load_data(
        data_folder=data_folder,
        exclude_file=region + ".csv",
        target=params["target"],
        cats=params["mw_cats"],
        rand_seed = None, # for sampling with replacement
        kernel_size=params["kernel_size"],
        use_cache=use_cache,
        protect_great=params["protect_great"]
    )

    # Define arguments to be passed into our training loop function.
    full_train_kwargs = dict(
        df=preprocessor.dataframe,
        inputs=preprocessor.inputs,
        model_name_add="test",
        hyperparam_config=hyperparam_config,
        use_wandb=False,
    )

    rand_seed = random.randint(0, 999999999)
    rand_seed = random.randint(0, 999999999)
    torch.manual_seed(rand_seed)

    fit_obj = Fit(**full_train_kwargs)


# In[9]:


if do_train:
    fig, ax = plt.subplots(dpi = 100)
    ax.plot(fit_obj.out_df["EPOCH"], fit_obj.out_df["TR_LOSS"], label = "Training loss")
    ax.plot(fit_obj.out_df["EPOCH"], fit_obj.out_df["VL_LOSS"], label = "Validation loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    min_vl = fit_obj.out_df["VL_LOSS"].min()

    print(fit_obj.out_df["TR_LOSS"].min())
    print(min_vl)
    print(fit_obj.out_df[fit_obj.out_df["VL_LOSS"] == min_vl])


# In[10]:


# enter epoch to use - use the index above 
# or the EPOCH value - 1
if do_train:
    epoch_use = np.where(fit_obj.out_df["VL_LOSS"] == min_vl)[0][0]
    print(epoch_use)


# In[11]:


# params, _ = convert_hyperparam_config_to_values(hyperparam_config) # convert here to include whatever were overriden
# data_suffix = params.get("dataset", "16k")  # which dataset you want as input
# data_folder = default.ROOT_DATA_DIRECTORY / data_suffix
# use_cache = False

# region = "sam"

# preprocessor = load_data(
#         data_folder=data_folder,
#         exclude_file= region + ".csv", #params["exclude_file"],
#         target=params["target"],
#         cats=params["mw_cats"],
#         rand_seed = None, # for sampling with replacement
#         kernel_size=params["kernel_size"],
#         skip_drop_na = True,
#         rd_exclude = False,
#         use_cache=use_cache,
#         protect_great=params["protect_great"]
#     )

# print(preprocessor.dataframe[preprocessor.dataframe["MW_CAT"] == 0].shape[0])
# print(preprocessor.dataframe[preprocessor.dataframe["MW_CAT"] == 1].shape[0])
# print(preprocessor.dataframe[preprocessor.dataframe["MW_CAT"] == 2].shape[0])
# print(preprocessor.dataframe.shape[0])


# In[12]:


params, _ = convert_hyperparam_config_to_values(hyperparam_config) # convert here to include whatever were overriden
data_suffix = params.get("dataset", "16k")  # which dataset you want as input
data_folder = default.ROOT_DATA_DIRECTORY / data_suffix
use_cache = False

dpi = 300 # use small if still testing

preprocessor = load_data(
        data_folder=data_folder,
        exclude_file= region + ".csv", #params["exclude_file"],
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
test_df = copy.deepcopy(pred_dl.dataset.dataframe)

out_df = {}
out_df["S_AVE"] = test_df["S_AVE"]
out_df["LON_AVE"] = test_df["LON_AVE"]
out_df["LAT_AVE"] = test_df["LAT_AVE"]
out_df["MR_ISC"] = test_df["MR_ISC"]
out_df["MR_GCMT"] = test_df["MR_GCMT"]
out_df["MW_CAT"] = test_df["MW_CAT"]

################## PERFORM PREDICTION 
model_fname = "epoch-" + str(epoch_use) + ".pt"
idx = 0

if make_map:

    if do_train:
        model_path = fit_obj.fit_folder/model_fname
    else:
        model_path = model_dir/folder_use/model_fname
        
    pred_obj = Fit(fit_on_init = False, **full_pred_kwargs, force_cats = 0) # initialize lang pirmi
    pred_model = copy.deepcopy(pred_obj.model)

    #loop through all 
    pred_model.load_state_dict(torch.load(model_path))
    pred_model.to(device)

    pred_model.eval()
    preds = np.zeros([len(pred_dl.dataset.dataframe), num_class])
    class_preds = np.zeros([len(pred_dl.dataset.dataframe)])

    # Loop through test data
    with torch.no_grad():
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
            #print(cat)

            preds[i] = pred_vals
            class_preds[i] = pred_vals2

        # save to dataframe for evaluation
        for i in range(num_class):
            out_df["MDL_"+ str(idx) + "_CLS_" + str(i)] = preds[:, i]

    out_df["MDL_" + str(idx) + "_PRED"] = class_preds
    idx += 1

    out_df = pd.DataFrame(out_df)
    out_df["CLASS_PRED"] = class_preds

    ########## prepare the results
    num_model = 1

    col_list =["MDL_" + str(x) + "_CLS_0" for x in range(num_model)]
    cls0_df = out_df[ ["S_AVE"] + col_list]
    cls0_df = cls0_df.assign(MEAN=cls0_df[col_list].mean(axis = 1))
    cls0_df = cls0_df.assign(STD=cls0_df[col_list].std(axis = 1))

    col_list = ["MDL_" + str(x) + "_CLS_1" for x in range(num_model)]
    cls1_df = out_df[["S_AVE"] + col_list]
    cls1_df = cls1_df.assign(MEAN=cls1_df[col_list].mean(axis = 1))
    cls1_df = cls1_df.assign(STD=cls1_df[col_list].std(axis = 1))


    col_list = ["MDL_" + str(x) + "_CLS_2" for x in range(num_model)]
    cls2_df = out_df[["S_AVE"] + col_list]
    cls2_df = cls2_df.assign(MEAN=cls2_df[col_list].mean(axis = 1))
    cls2_df = cls2_df.assign(STD=cls2_df[col_list].std(axis = 1))

    col_list = ["MDL_" + str(x) + "_PRED" for x in range(num_model)]
    pred_df = out_df[["S_AVE"] + col_list]

    ######### Create the map
    # just run every iteration since out_df is here - actually just need to run during first iteration
    workaround_df = workaround_df[workaround_df["S_AVE"] <= out_df["S_AVE"].max()]  
    lon_grid = lon_grid%360 # convert to 0 - 360

    for s in out_df["S_AVE"].unique():
        val_neg = workaround_df[(workaround_df["S_AVE"] == s) & (workaround_df["N_AVE"] < 0)] 
        val_pos = workaround_df[(workaround_df["S_AVE"] == s) & (workaround_df["N_AVE"] > 0)] 
        
        out_df.loc[out_df["S_AVE"] == s, "LON_NEG"] = val_neg["LON_AVE"].max()
        out_df.loc[out_df["S_AVE"] == s, "LAT_NEG"] = val_neg["LAT_AVE"].max()

        out_df.loc[out_df["S_AVE"] == s, "LON_POS"] = val_pos["LON_AVE"].max()
        out_df.loc[out_df["S_AVE"] == s, "LAT_POS"] = val_pos["LAT_AVE"].max()

    out_df = out_df.sort_values(by = "S_AVE", ascending = True)

    ########## Assign colors
    for category in out_df["MW_CAT"].unique(): # expected class
        out_df.loc[out_df["MW_CAT"] == category, "EX_CLR"] = color_list[category]

    for category in out_df["CLASS_PRED"].unique(): # predicted class
        out_df.loc[out_df["CLASS_PRED"] == category, "ML_CLR"] = color_list[int(category)]

    ''' create the maps containing the predictions '''

    mapper = vis_pkg.Mapper(dpi = dpi,
                            data = None,
                            topo = {'LON': lon_topo, 'LAT': lat_topo, 'VAL': topo_hs},
                            extent = other_settings["map_extent"],
                            dim_inch = other_settings["map_wh_inch"],
                            is_alu = is_alu,
                            is_ker = is_ker
                            )
    mapper.create_basemap()
    mapper.add_trench_line(trench_df = trench_plt, linewidth = 0.7)
    mapper.add_trench_marker(trench_fname = root_dir/conf["trench_used"],
                            trench_arr_sz = arr_sz
                            )
    mapper.add_topo()
    # plot for the predicted class
    mapper.add_ml_1d_data_poly(df = out_df, lon_col = "LON_POS", lat_col = "LAT_POS", clr_col = "ML_CLR", sort_by = sort_by, alpha = 1)
    # plot for the expected class
    mapper.add_ml_1d_data_poly(df = out_df, lon_col = "LON_NEG", lat_col = "LAT_NEG", clr_col = "EX_CLR", sort_by = sort_by, alpha = 1, 
                            poly_edge_width = 0.1)
    mapper.add_slip_outlines(eq_outlines_dict = {'OUTLINE': eq_outlines, 
                                                'COLOR': "mediumblue", 
                                                'AS_LINE':True})
    mapper.add_map_labels(label_dict = {'LAB': label_df, 'REL_POS': rel_pos})

    # the legend
    leg_lab = [ r"$M_w < 6.1$", 
                r"$6.1 \leq M_w < 8.3$",
                r"$M_w \geq 8.3$"]
    patch_list = [mpatches.Patch(color = x, ec = "k", linewidth = 0.2, label = "C" + str(i) + ": " + lab) for i, (lab, x) in enumerate(zip(leg_lab, color_list))]

    if region in ["kur"]:
        legend = mapper.ax.legend(handles=patch_list, fontsize = 7, loc = "upper left")
        legend.get_frame().set_linewidth(0.2)

    # save the files
    out_dir.mkdir(parents=True, exist_ok=True) 
    map_fname = out_dir/(region + "_rank_" + str(mod_rank) + ".png")
    print(map_fname)
    plt.savefig(map_fname, dpi = "figure", bbox_inches='tight')
    plt.close()


# In[ ]:





# In[13]:


preprocessor.cats


# In[14]:


# this will return the f1-score for class 0, 1, 2
f1_score(out_df["MW_CAT"], out_df["CLASS_PRED"], average = None) 


# In[15]:



''' SECTION ON XAI '''
# if ~make_map:
#     model_path = model_dir/use_model/model_fname # read the model again

model_name = region + "_rank_" + str(mod_rank)
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

    if use_tp: # only use true positives
        _, actual_cls = torch.max(torch.sigmoid(cat_labels), 1)
        add_cond = pred_vals2.item() == actual_cls.item()
        #print(add_cond)
    else:
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

## Plot the results of XAI analysis
for i in range(3):
    fig, ax = plt.subplots(dpi = 300, figsize = (0.7, 15))
    if do_stnd:
        sns.heatmap(all_median_df[[i]], cmap = cmap, vmin = 0, vmax = 3, linewidths = 1e-3, linecolor = "gray",
                    cbar_kws={'label': r"$\tilde{R}$", 
                                "shrink" : 3,
                                "location": "bottom", 
                                "orientation": "horizontal",
                                "pad": 0.02,
                                "ticks": [-3, -2, -1, 0, 1, 2, 3]})
    else:
        sns.heatmap(all_median_df[[i]], cmap = cmap, vmin = 0, vmax = 5, linewidths = 1e-3, linecolor = "gray",
                    cbar_kws={'label': r"$\tilde{R}$", 
                                "shrink" : 3,
                                "location": "bottom", 
                                "orientation": "horizontal",
                                "pad": 0.02,
                                "ticks": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]})
            

    # broken lines
    ax.hlines(borders, 0, 1, linestyle = "-", colors = "k", linewidth = 0.8)

    bnds = np.array([0] + borders + [borders[-1] + 3])
    bnds = 0.5*(bnds[1:] + bnds[:-1])

    ax.set_yticks(bnds)
    ax.set_yticklabels(xtick_lab, rotation = 0, fontsize = 10)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels([hmap_labels[i]],
                    rotation = -30, ha = "right")

    # save the heatmap
    if apply_thresh:
        plt_fname = out_dir/(model_name + "_thresh_" + str(thresh_val) + "_hmap_c" + str(i) + ".png")
    else:
            plt_fname = out_dir/(model_name + "_hmap_c" + str(i) + ".png")

    plt.savefig(plt_fname, dpi = "figure", bbox_inches='tight')
    plt.close()

    # histogram of the relevance values
    if not apply_thresh:
        plt.rc('font', size=10)
        fig, ax = plt.subplots(dpi = 300, figsize = (2, 2))

        n, bins, patches = ax.hist(all_median_df[[i]], bin_use, density=False, histtype='step',
                                cumulative=False, label='Empirical', linewidth = 1, color = "k")
        # n, bins, patches = ax.hist(all_median_df[[i]], bin_use, density=True, histtype='step',
        #                         cumulative=True, label='Empirical', linewidth = 1, color = "k")
        #ax.set_yscale("log")
        ax.set_ylim([0, 20])
        # ax.set_ylim([0, 1]) # for cumulative
        # ax.vlines(0.05, 0, 1, linestyles = '--', colors = "red", linewidth = 0.5)
        if do_stnd:
            ax.set_xlim([-3, 3])
        else:
            ax.set_xlim([-5, 5])
        ax.set_xlabel(r"$\tilde{R}$")
        ax.set_ylabel("Count")
        plt.rc('font', size=10)

        # save the histogram
        plt_fname = out_dir/(model_name + "_rhist_c" + str(i) + ".png")
        #plt_fname = out_dir/(model_name + "_cumu_rhist_c" + str(i) + ".png")
        plt.savefig(plt_fname, dpi = "figure", bbox_inches='tight')
        plt.close()


# In[16]:


print(all_median_df[[2]].min())
print(all_median_df[[2]].max())


# In[17]:


plt_fname


# In[18]:


add_cond

