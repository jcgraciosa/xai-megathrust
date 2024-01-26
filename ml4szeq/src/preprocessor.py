# -*- coding: utf-8 -*-
"""
All preprocessing related code here.
"""

import re
from functools import cached_property, lru_cache
from pathlib import Path
from typing import List, NewType, Tuple, Union
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer

import numpy as np
import pandas as pd
from tqdm import tqdm
import default

# Just a type hint for regex pattern strings
RegexPattern = NewType("RegexPattern", str)


class Preprocessor:
    """
    This class represents all the preprocessing that should be done on the
    data before being fed into our model for training/validation.
    """

    def __init__(
        self,
        data_folder: Path,
        exclude_file: str,
        quake_cat_list: List[Union[float, int]],
        target: str,
        protect_great: bool,
        noise_threshold: float = 0.0,
        rd_exclude: bool = False,
        rand_seed: int = None,
        skip_drop_na: bool = False,
        kernel_size: Tuple[int, int] = None,
        exclude_for_input: List[RegexPattern] = None,
        tr_half_use = None,     # added Dec 1, 2023.
        sep_dist = None,        # used in tr_half_use and tr_all_region
        tr_all_region = None,   # 
        tr_all_region_N = None  # valid values: 0, 1, 2, 3, 4

    ):
        # logic for tr_half_use:
        # Possible values: None - no effect; 
        # "first" - use first half for training;
        # "second" - use 2nd half for training;  
        # tr_all_region - should be None (ignored though)

        # logic for tr_all_region
        # set tr_all_region to True or False
        # make sure that rm_list is None (filename not found in list)
        # set sep_dist to a number

        # sep_dist is only used when tr_half_use is not None
        # this is the separation distance between the training data and test data
        # e.g. sep_dist = 1000 km and tr_half use is "first"
        # only S_AVE.min() up to (0.5*S_AVE.max() - 1000 km) are used for training

        self.data_folder = data_folder
        self.exclude_file = exclude_file
        self.rd_exclude =rd_exclude
        self.tr_half_use = tr_half_use
        self.sep_dist = sep_dist
        self.tr_all_region = tr_all_region
        self.tr_all_region_N = tr_all_region_N
        self.cats = quake_cat_list
        # minimum MW magnitude (not inclusive) to be considered an event
        self.noise_threshold = noise_threshold
        self.rand_seed = rand_seed
        self.kernel_size = kernel_size
        # Default to exclude certain variables to determine our actual desired input columns
        self.exclude_for_input = (
            exclude_for_input if exclude_for_input else default.EXCLUDE_FOR_INPUT
        )
        self.target = target
        self.protect_great = protect_great

        self.dataframe = pd.DataFrame()

        self.read_data_from_files()
        if skip_drop_na:
            self.drop_nas(0.1) # will likely not drop any columns in here
        else:
            self.drop_nas()
        self.normalise_inputs()
        self.quake_categoriser()

        # add option to do sampling with replacement
        # make sure that preprocessing eliminates the same amount of data 
        # so we do the sampling last
        if self.rand_seed is not None:
            self.dataframe = self.dataframe.sample( frac = 1.0, 
                                                    replace = True, 
                                                    random_state = self.rand_seed,
                                                    ignore_index = True)

        # remove according to the self.rm_list
        # logic for removing some of the datasets for training or testing
        if len(self.rm_list) > 0:
            if self.tr_half_use is None:
                
                print("Will remove all of the data in the rm_list ... ")
                print("remove list: ", self.rm_list)
                self.dataframe = self.dataframe[~self.dataframe["REGION"].isin(self.rm_list)]
            
            else: # tr_half_use is not None
                if self.rd_exclude: # doing TESTING so rm_list is more than 1 file
                    
                    # remove all files in rm_list
                    self.dataframe = self.dataframe[~self.dataframe["REGION"].isin(self.rm_list)] # remove all
                    
                    # then execute the halfing
                    if self.tr_half_use == "first": # training used first half
                        self.dataframe = self.dataframe[self.dataframe["S_AVE"] > 0.5*self.dataframe["S_AVE"].max()]
                    elif self.tr_half_use == "second": # training used second half
                        self.dataframe = self.dataframe[self.dataframe["S_AVE"] <= 0.5*self.dataframe["S_AVE"].max()]
                    
                    print(f"will remove {self.tr_half_use} half of {self.exclude_file} for testing ...")

                else: # doing training

                    cond1 = ~self.dataframe["REGION"].isin(self.rm_list) # True if NOT IN remove list
                    cond2 = self.dataframe["REGION"].isin(self.rm_list) # True if IN the remove list

                    if self.tr_half_use == "first": # training will use first half
                        cond3 = (self.dataframe["S_AVE"] <= (0.5*self.dataframe["S_AVE"].max() - self.sep_dist)) # belongs to first half
                    elif self.tr_half_use == "second": # training will use second half
                        cond3 = (self.dataframe["S_AVE"] > (0.5*self.dataframe["S_AVE"].max() + self.sep_dist)) # belongs to second half
                    
                    cond23 = cond2 & cond3 # get the AND of both boolean variables # True if in the remove list and in first half
                
                    self.dataframe = self.dataframe[cond1 | cond23]

                    print(f"will retain {self.tr_half_use} half of {self.rm_list} for training ...")

        elif self.tr_all_region is not None: # set that rm_list is None, but add another condition so that occurence of unknown behavior is reduced     
            
            #print(f"Length of dataset before removal: {self.dataframe.shape[0]}")
            region_list = self.dataframe["REGION"].unique()
            for i, reg in enumerate(region_list):

                min_s_ave = self.dataframe[self.dataframe.REGION == reg]["S_AVE"].min()
                max_s_ave = self.dataframe[self.dataframe.REGION == reg]["S_AVE"].max() 
                rng_s_ave = max_s_ave - min_s_ave

                # print(min_s_ave, max_s_ave)
                # print((min_s_ave + 0.2*self.tr_all_region_N*rng_s_ave - self.sep_dist))
                # print((min_s_ave + 0.2*(self.tr_all_region_N + 1)*rng_s_ave + self.sep_dist))

                if self.tr_all_region: # doing TRAINING using portions from all regions
                    # get boolean array to decide whether that row is included in training or not
                    cond_dist = (self.dataframe["S_AVE"] < (min_s_ave + 0.2*self.tr_all_region_N*rng_s_ave - self.sep_dist)) | \
                                (self.dataframe["S_AVE"] > (min_s_ave + 0.2*(self.tr_all_region_N + 1)*rng_s_ave + self.sep_dist))
                else: # do TESTING with portions from tr_all_region
                    # get boolean array to decide whether that row is included in training or not
                    if i < len(region_list) - 1: # not last
                        cond_dist = (self.dataframe["S_AVE"] >= (min_s_ave + 0.2*self.tr_all_region_N*rng_s_ave)) & \
                                    (self.dataframe["S_AVE"] < (min_s_ave + 0.2*(self.tr_all_region_N + 1)*rng_s_ave))
                    else:
                        cond_dist = (self.dataframe["S_AVE"] >= (min_s_ave + 0.2*self.tr_all_region_N*rng_s_ave)) & \
                                    (self.dataframe["S_AVE"] <= (min_s_ave + 0.2*(self.tr_all_region_N + 1)*rng_s_ave)) # <= is the difference
                              
                cond_dummy = cond_dist & (self.dataframe["REGION"] == reg)
                #print(cond_dummy.sum())

                if i == 0: # first
                    cond_fin = cond_dummy
                else:
                    cond_fin = cond_fin | cond_dummy
            self.cond_fin = cond_fin
            self.dataframe = self.dataframe[cond_fin]
       
        self.dataframe = self.dataframe.reset_index(drop = True)

    @cached_property
    def inputs(self):
        exclude = self.exclude_for_input
        # Find all columns, determine which ones we actually want to *include*
        columns = self.dataframe.columns.to_list()
        inputs = []

        # Loop over columns. For each column, test it against all regex patterns.
        # If it matches any pattern at all, then it will NOT be added to the list
        # of inputs
        for column in columns:
            include_flag = True
            for regex in exclude:
                if re.search(regex, column):
                    include_flag = False
                    break
            if include_flag:
                inputs.append(column)

        return inputs

    def read_data_from_files(self):
        """
        Reads and combines the data from the given files into a single dataframe,
        which is stored in `self.dataframe`. Adds REGION column which just stores
        integers from 0..(n-1), where n is the number of regions/files we're
        using.

        NOTE: This will search for all csv files in the `data_folder`. If you 
        want to exclude a region (which is not advised, since regions are used
        in an embedding layer), you will have to change this!
        """
        # Read csvs into multiple dataframes
        regional_dfs = []

        self.rm_list = []

        for file in self.data_folder.iterdir():
            if file.match("*.csv"): # Ignore any non-csv files!
                df = pd.read_csv(self.data_folder / file, index_col=None, header=0)
                df["REGION"] = len(regional_dfs) # Ensures zero-indexing, required for embedding layer
                
                fstr = str(file)
                str_split = fstr.split("/")
                last_elem = str_split[-1]
                reg_name = last_elem.replace(".csv", "")
                df["REGION_NAME"] = reg_name
                
                if not self.rd_exclude: # reading the TRAINING data - revised Sept 04, 2023
                    #cond_imp = file.match("ker.csv") | file.match("izu.csv") | file.match("cam.csv") | file.match("ryu.csv") # remove regions with no large EQs
                    #if file.match(self.exclude_file) | cond_imp:  # also remove areas with no large events
                    if file.match(self.exclude_file):  # list down region to remove
                        self.rm_list.append(len(regional_dfs))
                        print("will remove: ", file)

                else: # reading the TEST data
                    if not file.match(self.exclude_file):  # list down region to remove
                        self.rm_list.append(len(regional_dfs))
                        print("will remove: ", file)

                regional_dfs.append(df)

        self.dataframe = pd.concat(regional_dfs, axis=0, ignore_index=True)

        # Replace all 0s, NAs with some threshold value (for target variable only!)
        self.dataframe = self.dataframe.fillna(value={self.target: self.noise_threshold})
        self.dataframe[self.target] = self.dataframe[self.target].replace(
            {0: self.noise_threshold}
        )
        # replace all events <= 1 by 4
        self.dataframe.loc[self.dataframe[self.target] <= 1, self.target] = 4
        # add a column containing random data 
        #self.dataframe["RND_CTRL"] = np.random.normal(scale = 25, size = self.dataframe.shape[0]) 

        # Add this label for later, since data may be synthesised and we need
        # to make sure synthesised data doesn't end up in validation!
        self.dataframe["SYNTHETIC"] = False

    def drop_nas(self, thresh=0.9):
        """
        First drops any columns which are entirely filled with NAs. Then drops
        any rows which contain NAs in the remaining columns of interest. I.e.
        any NAs in input variables or output variables, but does not include
        anything like LON, LAT, etc. We could just drop these latter columns,
        but we leave them in the dataframe in case we do want to use that
        information after training/validation to do any visualisations.
        """
        # Store original number of rows, cols to see how many are dropped
        start_rows, start_cols = self.dataframe.shape

        # For MW target variable, there are so few great earthquakes that we
        # want to protect them all. This means any potential input column 
        # which contains an NA in a great earthquake row *needs* to be dropped
        # entirely. Otherwise, we'll lose that great earthquake row later on!
        if self.protect_great:
            great_thresh = self.cats[-2] # Presumably where we start "great earthquakes" from
            great_earthquakes = self.dataframe[(self.dataframe["MW"] >= great_thresh)]
            # This probs isn't the best way of doing it, but wasn't sure how to
            # get dropna() to do what I wanted here.
            bad_cols_bool = (great_earthquakes.isna().sum(axis=0) > 0)
            bad_cols = self.dataframe.columns[bad_cols_bool]
            self.dataframe = self.dataframe.drop(columns=bad_cols)
            print("The following columns were dropped because they are NA in great earthquake rows:\n", bad_cols)

        # Some columns only contain a few non-NA values. We set a threshold so
        # we drop these columns that are mostly NAs. 
        minimum_rows = thresh * len(self.dataframe)
        print(f"Dropping any columns which have fewer than {thresh * 100: .0f}% ({minimum_rows}) values")
        self.dataframe = self.dataframe.dropna(axis="columns", thresh=minimum_rows)
        # Now drop any rows which contain *any* NAs (just in columns of interest)
        cols = self.inputs + [self.target, "REGION"]
        print("Dropping any rows which are missing *any* input variables")
        self.dataframe = self.dataframe.dropna(axis="rows", subset=cols, how="any")

        final_rows, final_cols = self.dataframe.shape
        print(
            f"Dropped {start_cols - final_cols} columns, and {start_rows - final_rows} rows."
        )

    @lru_cache(maxsize=1)
    def get_rounding_error(self):
        """
        Returns (min_S, min_N), estimated step sizes of data along S and N
        coordinates.

        E.g., for 2k data you should get (50, 50), and for 16k data you should
        get (25, 25).
        """
        # Get positive coords only (simplifies finding min) and discard anything
        # less than 1 (i.e. scrap S~0, which sometimes seems to have decimal
        # places after it!)
        df = self.dataframe

        min_S = min(df.loc[df["S_AVE"] > 1, "S_AVE"])
        min_S2 = min(df.loc[df["S_AVE"] > min_S, "S_AVE"])
        min_N = min(df.loc[df["N_AVE"] > 1, "N_AVE"])
        min_N2 = min(df.loc[df["N_AVE"] > min_N, "N_AVE"])
        ds = min_S2 - min_S
        dn = min_N2 - min_N

        return (ds, dn)

    def quake_categoriser(self):
        """
        Creates MW_CAT column in dataframe which stores integers from 0..(n-1),
        where n represents the number of categories we're looking at. The
        categories are determined by the list `self.cats`, and each category
        *includes* its lower bound and *excludes* its upper bound.
        """
        for i in range(len(self.cats) - 1):
            low_bound, high_bound = self.cats[i : i + 2]
            if i < len(self.cats) - 1 - 1:
                self.dataframe.loc[
                    (self.dataframe[self.target] >= low_bound)
                    & (self.dataframe[self.target] < high_bound),
                    ["MW_CAT"],
                ] = i
            else:
                 self.dataframe.loc[
                    (self.dataframe[self.target] >= low_bound)
                    & (self.dataframe[self.target] <= high_bound),
                    ["MW_CAT"],
                ] = i

        #print(self.dataframe["MW_CAT"])
        self.dataframe["MW_CAT"] = self.dataframe["MW_CAT"].astype("int")

    def find_neighbours(self):
        length = self.kernel_size[0]
        width = self.kernel_size[1]

        df = self.dataframe  # not copying, just shortening variable name
        all_neighbour_indices = [None for _ in range(len(df))]
        all_edge_bools = [False for _ in range(len(df))]

        print("Finding neighbours, adding NEIGHBOURS and IS_EDGE to dataframe...")
        for i, point in tqdm(df.iterrows(), total=len(df)):
            s, n, region = point["S_AVE"], point["N_AVE"], point["REGION"]
            neighbour_indices = pd.Int64Index([])
            # Default to false, but may get set to true later on
            is_edge = False

            # TODO: Might be quicker to use itertools.product instead of
            # nested for loop. Instead of computing 9 different boolean indices,
            # only compute one (that checks for 9 different coordinate pairs).
            # Can eaisly check if point on edge or not but seeing if it has less
            # than 9 indices in its neighbors array.
            for s in np.linspace(s - self.s_offset, s + self.s_offset, num=length):
                for n in np.linspace(n - self.n_offset, n + self.n_offset, num=width):
                    index_bool = (
                        (df["S_AVE"] == s)
                        & (df["N_AVE"] == n)
                        & (df["REGION"] == region)
                    )
                    # Convert to actual numeric index (much more space-efficient
                    # than having a series of however many thousand Falses with
                    # a single True nestled in there!)
                    index = df.index[index_bool]

                    # NOTE: Will be empty if no neighbour found
                    if index.empty:
                        is_edge = True
                    else:
                        neighbour_indices = neighbour_indices.append(index)

            # NOTE: This assumes index labels in dataframe start from 0 and go
            # in increasing order without any gaps. If we have any index labels
            # which are bigger than length of dataframe, ISSUES!
            all_neighbour_indices[i] = neighbour_indices
            all_edge_bools[i] = is_edge

        self.dataframe = self.dataframe.assign(
            NEIGHBOURS=all_neighbour_indices, IS_EDGE=all_edge_bools
        )

    def normalise_inputs(self):
        """
        Normalises all input variables (based on `self.inputs`) to a mean of 0,
        stdev of 1.
        Modified to robust scaler
        Modified to power transformer to produce Gaussian distribution
        """
        df = self.dataframe

        #df[self.target] = np.log10(df[self.target])
        #######

        # perform uniformization of the target variable - for regression
        self.qt = QuantileTransformer(random_state = 0)
        to_uniformize = np.array(df[self.target]).reshape(-1, 1)
        uni = self.qt.fit_transform(to_uniformize) 
        # replace the target with the uniform data
        df[self.target] = uni.flatten()
        ####### 

        # add inputs that are correlated with the target value
        # then add some noise
        # don't add feature that's completely the same as target as the model may focus on this
        # df["TRG_STD1"] = uni.flatten() + np.random.normal(scale = 1, size = uni.flatten().shape[0]) 
        # df["TRG_STD2"] = uni.flatten() + np.random.normal(scale = 2, size = uni.flatten().shape[0]) 
        # df["TRG_STD3"] = uni.flatten() + np.random.normal(scale = 3, size = uni.flatten().shape[0]) 
        # df["TRG_STD4"] = uni.flatten() + np.random.normal(scale = 4, size = uni.flatten().shape[0]) 
        # df["TRG_STD5"] = uni.flatten() + np.random.normal(scale = 5, size = uni.flatten().shape[0]) 

        # self.inputs = self.inputs + ["TRG_STD1", "TRG_STD2", "TRG_STD3", "TRG_STD4", "TRG_STD5"]  
        inputs = self.inputs

        print("Executing power transformer ... ")
        scaler = PowerTransformer() 
        df[inputs] = scaler.fit_transform(df[inputs])

        # standardized scaling
        # mean = df[inputs].mean()
        # std = df[inputs].std()
        # df[inputs] = (df[inputs] - mean) / std
