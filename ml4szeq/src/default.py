from pathlib import Path

from utils import get_config

############ MODEL/PREPROCESSING PARAMETERS WHICH SHOULDN'T CHANGE ############
# Since we have so many columns in our newer datasets, it would be silly to whitelist
# all columns that we want to use as input variables. Therefore, we create this
# blacklist below. All of these are treated as regex pattersn, which allows us
# to easily exclude some repetitive patterns in columns (for example, any
# percentiles for all the different geophysical characteristics).
EXCLUDE_FOR_INPUT = [
    "^LON$",    # search beginning and end
    "^LON_AVE$",
    "^LAT_AVE$",
    "^S_AVE$",
    "^N_AVE$",
    "^LAT$",
    "^DEPTH$",
    "^DATE$",
    "^TIME$",
    "^MR",  # We have a few different MR_ columns in some datasets
    "^MW",  # We have a few different MW_ columns in some datasets
    "SYNTHETIC", # This just represents whether data has been synthesised or not
    "REGION",
    "NEIGHBOURS",
    "IS_EDGE",
    "NEAREST",
    "DEP_AVE",  # Only DEP variable.
    # VARIABLES WE WERE TOLD TO IGNORE
    "^THK",  # Starts with THK
    "^STR",  # Starts with STR
    "^DIP",  # Starts with DIP,
    # PERCENTILES ($ means they are at END of string)
    "P05$",
    "P25$",
    "P50$",
    "P75$",
    "P95$",
    "_PKF$", # remove these as they sometimes have 0 values 
    "_RMS$",
    "_EN1$",
    "_EN2$",
    # to remove for scenario 3
    "_KUR$",
    "_150$",
    "_800$",
    "_1000$",
    "^FD2_",
    "^FM2_",
    "^SD2_",
    "^SM2_",
    ####### end to remove for scenario 3
    # additional items to remove for scenario 4
    # "_400$",
    # "_SKW$",
    ####### end to remove for scenario 4
    "^CRD_DG_", # removes a lot of data due to blanks if included
    "^CRS_DG_", # search beginning
    "^CRM_DG_",
    "^SRO_UP_",
    "^IRO_UP-",
    "^LRO_UP_",
    "^RBA_UP",
    "^RBA_DG",
    "^RGR_UP",
    "^RGR_DG",
    ##### remove for final analysis - see excel for reasoning
    "^INV_DG_",
    "^DLT_DG_",
    "^FD1_",
    "^FM1_",
    "^FDL_",
    "^SD1_",
    "^SM1_",
    "^SDL_",
    "^BGR_DG_",
    "^EGO_DG_",
    "^EGO_L_DG_",
    "^EGO_SL_DG_",
    "^EGO_UM_DG_",
    "^EGR_L_DG_",
    "^EGR_L_UP_",
    "^EGR_SL_DG_",
    "^EGR_SL_UP_",
    "^EGR_UM_DG_",
    "^EGR_UM_UP_",
    "^EGR_BG_DG_",
    "_SKW$",
    "_MIN$",
    "_MAX$",
    "_RNG$"
]
 
EXCLUDE_FOR_INPUT = EXCLUDE_FOR_INPUT

################################## FILEPATHS ##################################
# Root folder set as described in config. Defaults to current working dir.
ROOT_DIRECTORY = Path(get_config("ROOT_DIRECTORY", Path.cwd()))
# Directory in which we store all of our output files/sub-directories.
ROOT_OUTPUT_DIRECTORY = ROOT_DIRECTORY / "out"
MODELS_DIRECTORY = ROOT_OUTPUT_DIRECTORY / "models/phys_transition/sum" # always use scenario3
PREDICTIONS_DIRECTORY = ROOT_OUTPUT_DIRECTORY / "predictions"
HEAT_MAPS_DIRECTORY = ROOT_OUTPUT_DIRECTORY / "heat_maps"

# The root directory where we read all of our data from. This should include
# subdirectories for the various datasets we have (e.g., "2k", "16k",
# "16k_more_vars").
ROOT_DATA_DIRECTORY = ROOT_DIRECTORY / "data"
# Folder to store parameter yaml files for W&B sweeps
WANDB_PARAMETERS_DIRECTORY = ROOT_DIRECTORY / "parameters"
