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

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.std import tqdm

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import vis_pkg
import helper_pkg
import json
import cartopy as cartopy
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["font.family"] = "Arial"
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3

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
params, _ = convert_hyperparam_config_to_values(hyperparam_config)


# In[2]:


'''
Updated definition of LRP function in here
'''
# from model import * 

# allowed rule_list keys
# "zero" - LRP - 0 for the output layer
# "eps" - LRP - e for the hidden layers
# "w2" - LRP - w2 rule for the input  
# Note: rule_list - start from the input to the output

def rho(w,val):  
    # [None,0.1,0.0,0.0][l] - indexing of the list then multiplied to the activation function
    return w + val * torch.maximum(torch.zeros_like(w), w)

def incr(z,val): 
    return z + val * ((z**2).mean()**.5) + 1e-9

def compute_relevance(pred_dl, len_pred_dl, pred_model, n_input, rho_list = [0., 0.1, 0., 0.], incr_list = [0., 0., 0.1, 0.], use_w2 = True):

    # dis-assemble the model - outside data loop
    layers = [module for module in pred_model.modules() if isinstance(module, FC)]
    layers = layers + [nn.Linear(hidden[-1], n_out)]

    num_layers = len(layers)
    relevance_arr = np.zeros([len_pred_dl, n_input])

    with torch.no_grad():
        for j, ((x_cont, x_region), (cat_labels, cont_labels)) in enumerate(pred_dl): 
            x_cont, x_region, cat_labels, cont_labels = (
                                                            x_cont.to(device),
                                                            x_region.to(device),
                                                            cat_labels.to(device),
                                                            cont_labels.to(device),
                                                        )

            X_in = torch.cat([x_cont], 1).type(torch.float32)

            activ = [X_in] + [X_in]*num_layers # activations

            # forward propagation
            for k in range(num_layers):
                #print("layer: ", i)
                activ[k + 1] = layers[k].forward(activ[k])

            # prepare relevance of the layers
            T = activ[-1].detach() # detach as separate entity
            index = T.abs().max(1).indices
            Tmax = T.abs().max(1).values
            T = torch.FloatTensor(np.array(T.abs())*0)
            T[np.arange(T.shape[0]), index] = Tmax
            #print(T)
            R = [None]*num_layers + [T] # relevance for the layers (only outmost is populated)
            #print(len(R))
            

            # calculate relevance for all layers
            for l in range(0, num_layers)[::-1]:

                # old code 
                #rule = rule_list[l]
                # # what is the purpose of using operation
                # # just returns the val added with something little 
                # if rule == "w2":
                #     rho_trans = lambda val, fac : torch.square(val)
                # else:
                #     rho_trans = lambda val, fac : val + fac*val.clamp(min = 0) # val must be a tensor
                # end of old code

                #print("current l value: ", l)
                layer = copy.deepcopy(layers[l])

                if l > 0:
                    if isinstance(layer, FC): 
                        layer.fc.weight = nn.Parameter(rho(layer.fc.weight, rho_list[l]))
                        layer.fc.bias = nn.Parameter(rho(layer.fc.bias, rho_list[l])) if layer.fc.bias is not None else None
                    else:
                        layer.weight = nn.Parameter(rho(layer.weight, rho_list[l]))
                        layer.bias = nn.Parameter(rho(layer.bias, rho_list[l]))
                else: # input layer
                    if use_w2:
                        if isinstance(layer, FC): 
                            layer.fc.weight = nn.Parameter(torch.square(layer.fc.weight) + 1e-9)
                            layer.fc.bias = nn.Parameter(torch.square(layer.fc.bias) + 1e-9) if layer.fc.bias is not None else None
                        else:
                            layer.weight = nn.Parameter(torch.square(layer.weight) + 1e-9)
                            layer.bias = nn.Parameter(torch.square(layer.bias) + 1e-9)
                    else:
                        pass # do nothing

                activ[l] = activ[l].data.requires_grad_(True) # is this really needed?

                # step 1
                # old code
                # if rule == "zero":
                #     z_val = layer.forward(activ[l])
                # if rule == "eps":
                #     z_val = layer.forward(activ[l]) + eps_val
                # elif rule == "w2":
                #     ones_tensor = torch.ones_like(activ[l])
                #     z_val = layer.forward(ones_tensor)

                if l > 0:
                    z_val = incr(layer.forward(activ[l]), incr_list[l])
                else: # input layer
                    if use_w2:
                        ones_tensor = torch.ones_like(activ[l])
                        z_val = layer.forward(ones_tensor)
                    else:
                        w = layer.fc.weight # sure that it has fc
                        wp = torch.maximum(torch.zeros_like(w), w)
                        wm = torch.maximum(torch.zeros_like(w), w)
                        lb = activ[l]*0. - 1.
                        hb = activ[l]*0. + 1.
                        z_val = torch.matmul(activ[l], torch.transpose(w, 0, 1)) -                                 torch.matmul(lb, torch.transpose(wp, 0, 1)) -                                 torch.matmul(hb, torch.transpose(wm, 0, 1)) -                                 +1e-9
        

                # step 2
                s_val = (R[l + 1]/z_val).data
            
        
                # step 3 - for linear layers
                if l > 0:
                    if isinstance(layer, FC): 
                        c_val = torch.matmul(s_val, layer.fc.weight)
                    else:
                        c_val = torch.matmul(s_val, layer.weight)
                else:
                    if use_w2:
                        c_val = torch.matmul(s_val, layer.fc.weight) # sure that it has fc
                    else:
                        c, cp, cm = torch.matmul(s_val, w), torch.matmul(s_val, wp), torch.matmul(s_val, wm)  

                # # step 4
                if l > 0:
                    R[l] = (activ[l]*c_val).data
                else:
                    if use_w2:
                        R[l] = (ones_tensor*c_val).data
                    else:
                        R[l] = activ[l]*c-lb*cp-hb*cm    
                
            # save R[0]
            relevance_arr[j] = R[0].numpy()

    return relevance_arr


# In[3]:


data_suffix = params.get("dataset", "16k")  # which dataset you want as input
data_folder = default.ROOT_DATA_DIRECTORY / data_suffix
use_cache = False
print(f"Preprocessing data ({use_cache=})...")

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
print(f"Finished preprocessing!")


# In[4]:


# for val in preprocessor.inputs:
#     print(val)

print(len(preprocessor.inputs))


# In[5]:


# Define arguments to be passed into our training loop function.
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


# In[6]:


model_dir = Path("/Users/jgra0019/Documents/codes/ml4szeq/ml4szeq/out/models/sum/sum_300_500_1e-02_2022-04-26T18:57:51.355504")
epoch_use =15 # decide which epoch to use based on trval loss
num_class = 5
num_model = 1
device = "cpu"
region = "sum"

model_fname = "epoch-" + str(epoch_use) + ".pt"
idx = 0

model_path = model_dir/model_fname
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


# In[7]:


'''
LRP in here
'''
hidden = params['hidden_layers']
n_out = len(params["mw_cats"]) - 1 # with reference to model
n_input = len(pred_dl.dataset.inputs) # with reference to model 
len_pred_dl = len(pred_dl.dataset.dataframe)
eps_val = 1e-6
key_use = region + "*"

relevance_3d_arr = np.empty((len_pred_dl, n_input))

# inside model loop

model_path = model_dir/model_fname
pred_obj = Fit(fit_on_init = False, **full_pred_kwargs) # initialize lang pirmi
pred_model = copy.deepcopy(pred_obj.model)

#loop through all 
pred_model.load_state_dict(torch.load(model_path))
pred_model.to(device)

pred_model.eval()

rel_2d_arr = compute_relevance(pred_dl, len_pred_dl, pred_model, n_input, use_w2 = True) # use default rules

# normalize along the rows
norm_fac = rel_2d_arr.sum(axis = 1)
relevance_3d_arr = 100*rel_2d_arr/norm_fac[:, None]



# # Plotting the results 

# ## Plotting the predictions

# In[8]:


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


# In[9]:


''' 
Map of the predictions
This might not be useful in the future 
'''

conf_dir = Path("/Users/jgra0019/Documents/codes/region-paths")
conf_fname = conf_dir/(region + "_conf.json")
map_set_fname = "/Users/jgra0019/Documents/codes/globdat-paths/map_setting_conf.json"

with open(conf_fname) as json_data_file:
    conf = json.load(json_data_file)

with open(map_set_fname) as data_file:
    print(map_set_fname)
    all_map_setting = json.load(data_file)

map_setting = all_map_setting[region]
root_dir = Path(conf["mac_root"])
other_settings = conf["conf"]
conf = conf["path"]

##########################
# creation of the map
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
label_df.loc[label_df["COLOR"] == 'yellow', "COLOR"] = 'firebrick'

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
n_max = 150 # in km in direction of downgoing plate
n_min = -150 # in km in direction of upper plate
dn = 150 # step in the n-axis
ds = 50 # step in the s-axis 

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
workaround_df = workaround_df[workaround_df["S_AVE"] <= out_df["S_AVE"].max()]
# convert lon_grid to 0 to 360
lon_grid = lon_grid%360
print("Done generating grid ...")


# In[10]:


for s in out_df["S_AVE"].unique():
    val_neg = workaround_df[(workaround_df["S_AVE"] == s) & (workaround_df["N_AVE"] < 0)] 
    val_pos = workaround_df[(workaround_df["S_AVE"] == s) & (workaround_df["N_AVE"] > 0)] 
    
    out_df.loc[out_df["S_AVE"] == s, "LON_NEG"] = val_neg["LON_AVE"].max()
    out_df.loc[out_df["S_AVE"] == s, "LAT_NEG"] = val_neg["LAT_AVE"].max()

    out_df.loc[out_df["S_AVE"] == s, "LON_POS"] = val_pos["LON_AVE"].max()
    out_df.loc[out_df["S_AVE"] == s, "LAT_POS"] = val_pos["LAT_AVE"].max()

out_df = out_df.sort_values(by = "S_AVE", ascending = True)


# In[11]:


# create a column for the colors 
color_list = ["C0", "C1", "C2", "C3", "C4"]
for category in out_df["MW_CAT"].unique(): # expected class
    out_df.loc[out_df["MW_CAT"] == category, "EX_CLR"] = color_list[category]

for category in out_df["CLASS_PRED"].unique(): # predicted class
    out_df.loc[out_df["CLASS_PRED"] == category, "ML_CLR"] = color_list[int(category)]

# setting for the linewidth
if region in ["alu", "sam"]:
    lwidth = 4
else:
    lwidth = 4

# setting for the sort_by
if region in ["alu"]:
    sort_by = 0
elif region in ["sam", "kur", "sum"]:
    sort_by = 1
else:
    sort_by = 1


# In[12]:


# create the maps
dpi = 300 # use small if still testing
mapper = vis_pkg.Mapper(dpi = dpi,
                        data = None,
                        topo = None,
                        extent = other_settings["map_extent"],
                        dim_inch = other_settings["map_wh_inch"],
                        is_alu = is_alu,
                        is_ker = is_ker
                        )
mapper.create_basemap()
mapper.add_trench_line(trench_df = trench_plt)
mapper.add_trench_marker(trench_fname = root_dir/conf["trench_used"],
                        trench_arr_sz = arr_sz
                        )

# plot for the expected class
mapper.add_ml_1d_data_poly(df = out_df, lon_col = "LON_POS", lat_col = "LAT_POS", clr_col = "EX_CLR", sort_by = sort_by, hatch = "....", alpha = 1)
# plot for the predicted class
mapper.add_ml_1d_data_poly(df = out_df, lon_col = "LON_NEG", lat_col = "LAT_NEG", clr_col = "ML_CLR", sort_by = sort_by)

# the legend
leg_lab = [ r"$M_w < 5.1$", 
            r"$5.1 \leq M_w < 6.6$",
            r"$6.6 \leq M_w < 8.2$",
            r"$8.2 \leq M_w < 8.5$", 
            r"$M_w \geq 8.5$"]
patch_list = [mpatches.Patch(color = x, label = "C" + str(i) + ": " + lab) for i, (lab, x) in enumerate(zip(leg_lab, color_list))]

#if region not in ["izu"]:
mapper.ax.legend(handles=patch_list, fontsize = 7)


# ## Check the accuracy
# But this might not be a good metric since there are still insights to be had even if accuracy is low

# In[13]:


f1_score_test = np.zeros(num_model)
acc_test = np.zeros(num_model)

for i in range(num_model):
    conf_mtrx = confusion_matrix(y_true=out_df["MW_CAT"], y_pred=out_df["MDL_" + str(i) + "_PRED"])

conf_mtrx


# ## Plotting the LRP results

# In[14]:


# for one model:
# 1. Calculate the total relevance for each column
# 2. Get the top and lowest 10
# 3. Then create a bar plot of the top 10 and lowest 10

class_use = 4

# just use one model
rel_one_model = relevance_3d_arr # get the relevance of just one model
 
tot_rel_df = {}
col_name_list = []
tot_rel_list = []
ave_rel_list = []
std_rel_list = []

cond = (out_df["MW_CAT"] == 4) & (out_df["MDL_0_PRED"] == 4)

for idx, colname in enumerate(pred_dl.dataset.inputs):

    col_name_list.append(colname)
    tot_rel_list.append(rel_one_model[cond, idx].sum())
    ave_rel_list.append(rel_one_model[cond, idx].mean())
    std_rel_list.append(rel_one_model[cond, idx].std())

tot_rel_df = {  "COLNAME": col_name_list, 
                "TOT_REL": tot_rel_list,
                "AVE_REL": ave_rel_list,
                "STD_REL": std_rel_list}

tot_rel_df = pd.DataFrame.from_dict(tot_rel_df)
tot_rel_df = tot_rel_df.sort_values(by = "TOT_REL", ascending = False)
# tot_rel_df  = tot_rel_df.reset_index(inplace = False, drop = True) Indices will be used later


# In[15]:


for i, col in enumerate(tot_rel_df["COLNAME"]):
    print(i, col)


# ## Volin plots of the relevance

# In[16]:


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.yaxis.set_tick_params(direction='out')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_ylim(0.25, len(labels) + 0.75)
    #ax.set_xlabel('Sample name')

def make_violin_plot(ax, data, labels, facecolor):
    parts = ax.violinplot(  data,    
                            showmeans = False, 
                            showmedians = False, 
                            vert = False, 
                            showextrema = False)
    for pc in parts['bodies']:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
        pc.set_alpha(0.5)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers = np.array([
        adjacent_values(np.sort(sorted_array), q1, q3) for sorted_array, q1, q3 in zip(data.T, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    #ax.scatter(medians, inds, marker='s', color='white', s=30, zorder=3)
    ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
    ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=0.7)

    # variable labels
    for ax in [ax]:
        set_axis_style(ax, labels)

    #plt.xticks(rotation = '30', ha = "right")
    #ax.set_xlabel("Variable")
    ax.set_xlabel(r"R")


# In[17]:


''' PLOT OF THE TOP 15 - VIOLIN PLOTS '''
tot_rel_df = tot_rel_df.sort_values(by = "AVE_REL", ascending = False)
# tot_rel_df  = tot_rel_df.reset_index(inplace = False, drop = True) The original indices are used later

sset_df = tot_rel_df[:10]
idx_use = np.array(sset_df.index)

fig, ax = plt.subplots(dpi = 150)

make_violin_plot(ax, 
                rel_one_model[:, idx_use], 
                np.array(pred_dl.dataset.inputs)[idx_use.astype(int)], 
                facecolor = "C0"
                )
#ax.set_xlim(-10, 10)


# In[18]:


'''PLOT OF THE LAST 15'''
sset_df = tot_rel_df[-10:]

idx_use = np.array(sset_df.index)

fig, ax = plt.subplots(dpi = 150)

make_violin_plot(ax, 
                rel_one_model[:, idx_use], 
                np.array(pred_dl.dataset.inputs)[idx_use.astype(int)], 
                facecolor = "C1"
                )
#ax.set_xlim(-10, 10)


# In[19]:


'''PLOT HISTOGRAMS OF THE MEAN RELEVANCE VALUES'''

fig, ax = plt.subplots(dpi = 600)
ax.hist(tot_rel_df["AVE_REL"], bins = 20, edgecolor = "black", linewidth = 0.5, color = "C3", alpha = 0.7)
ax.set_xlabel(r"$<R>$")
ax.set_ylabel("Counts")


# In[32]:


''' try to use the LRP from captum '''
from captum.attr import LRP

# still use class_use for now
# do LRP if predicted class and actual class are both equal to class_use

# maybe I need to read the model again
model_path = model_dir/model_fname
pred_obj = Fit(fit_on_init = False, **full_pred_kwargs, force_cats = 0) # initialize lang pirmi
pred_model = copy.deepcopy(pred_obj.model)

#loop through all 
pred_model.load_state_dict(torch.load(model_path))
pred_model.to(device)

pred_model.eval()

class_use = 4

#lrp = LRP(wrapped_model)
lrp = LRP(pred_model)
lrp_attr = None
conv_delta = None

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
    #print(cat)

    if (pred_vals2 == class_use) and (torch.argmax(cat_labels) == class_use):
        attribution, conv = lrp.attribute( x_cont, 
                                            target = torch.argmax(cat_labels).item(), 
                                            return_convergence_delta = True)

    if lrp_attr is None:
        lrp_attr = attribution
        conv_delta = conv
    else:
        lrp_attr = torch.cat((lrp_attr, attribution), axis = 0)
        conv_delta = torch.cat((conv_delta, conv), axis = 0)

        
cap_mean_out = lrp_attr.mean(axis = 0).detach().numpy()
cap_std_out = lrp_attr.std(axis = 0).detach().numpy()

lrp_attr = lrp_attr.detach().numpy()
conv_delta = conv_delta.detach().numpy()

# create the necessary dataframes in here
# one for the mean and standard deviation
# another one for the relevance values for each feature and sample
cap_mean_rel_df = {"FEATURE": pred_dl.dataset.inputs,
                   "AVE_REL": cap_mean_out,
                   "STD_REL": cap_std_out}

cap_rel_df = {}
for i, feat in enumerate(pred_dl.dataset.inputs):
    cap_rel_df[feat] = lrp_attr[:, i]

cap_mean_rel_df = pd.DataFrame.from_dict(cap_mean_rel_df)
cap_rel_df = pd.DataFrame.from_dict(cap_rel_df)


# cat

# In[33]:


fig, ax = plt.subplots(dpi = 100)
ax.errorbar(cap_mean_rel_df.index, cap_mean_rel_df["AVE_REL"], yerr = cap_mean_rel_df["STD_REL"],  ecolor = "k", elinewidth = 0.7, capsize = 3, marker = "o", markersize = 1, linewidth = 0, color = "C1")
ax.hlines(y = 0, xmin = -0.1, xmax = 2.1, color = "r", linestyle = "--")
#ax.set_xlim([-0.1, 2.1])


# In[34]:


fig, ax = plt.subplots(dpi = 600)
ax.hist(cap_mean_rel_df["AVE_REL"], bins = 50, edgecolor = "black", linewidth = 0.5, color = "C3", alpha = 0.7)
ax.set_xlabel(r"$<R>$")
ax.set_ylabel("Counts")
ax.set_yscale("log")


# In[53]:


cap_mean_rel_df["AVE_REL"].std()


# In[52]:


''' Create violin plots for the features with the LARGEST mean relevance '''
cond = (cap_mean_rel_df["AVE_REL"] > 0.18)
feat_use = cap_mean_rel_df[cond]["FEATURE"] 

fig, ax = plt.subplots(dpi = 150)

make_violin_plot(ax, 
                np.array(cap_rel_df[feat_use]), 
                np.array(feat_use), 
                facecolor = "C0"
                )


# In[50]:


''' Create violin plots for the features with the most NEGATIVE mean relevance '''
cond = (cap_mean_rel_df["AVE_REL"] < -0.2)
feat_use = cap_mean_rel_df[cond]["FEATURE"] 

fig, ax = plt.subplots(dpi = 150)

make_violin_plot(ax, 
                np.array(cap_rel_df[feat_use]), 
                np.array(feat_use), 
                facecolor = "C1"
                )


# In[ ]:


# how to proceed?
# try different rules? for simplicity, use the same rules in the entire network
# use integrated gradients
# how about the false positives and false negatives - these will help build the story
# plot the features and see how they vary along trench 
# think why these are relevant - part of interpretation
# how to automate? - do this to make things faster


# In[54]:


cap_rel_df.shape


# In[ ]:




