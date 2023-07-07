import math
from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index
import torch
import wandb
from imblearn.over_sampling import SMOTENC
from numpy.random import default_rng
from torch import nn, optim

import default
from dataset import DFDataset
from model import EarthquakeMTL
from train import train_loop
from utils import convert_hyperparam_config_to_values, reset_weights
from validate import validation_loop


# A random number generator from numpy
rng = default_rng()


class Fit:
    """
    A class representing the creation and fitting of a model to a dataframe.
    When we say fitting here, we mean both training and validation.
    """
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # How much of data should be used for validation (if not using k-fold)
    # TODO: Consider making this related to number of k-folds set in parameters,
    # even if k-folds not being used. E.g., 5 k-folds = 0.2, 10 k-folds = 0.1.
    # This would be pretty unobvious behvaiour though.
    VALIDATION_FRAC = 0.2

    def __init__(
        self,
        df: pd.DataFrame,
        inputs: List[str],
        hyperparam_config: dict,
        model_name_add: str,
        use_wandb: bool = False,
        fit_on_init: bool = True,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        df: pandas.DataFrame
            The preprocessed dataframe containing all variables of interest (
                inputs, regions, target categories).
        inputs: List[str]
            List of labels of columns to use from `df` as inputs to the model.
            Excludes "REGION", since this is an entirely separate input.
        hyperparam_config: dict
            A hyperparameter config dictionary as described in W&B's 
            documentation. See their yaml file specifications for more details.
            NOTE: We only expect one value per parameter for this class (if you
            want to test different combinations, you must either do it by hand,
            or code the combinations yourself. This was implemented in the past
            using Cartesian products, but was pretty hideous to look at and not 
            useful enough!).
        use_wandb: bool
            Whether to log to W&B or not.
        fit_on_init: bool
            Whether to immediately fit model to data upon initialisation or not.
            Setting this to False could be useful if you want to load a state 
            dictionary on the model before fitting, for example.
        """
        self.df = df
        self.inputs = inputs
        self.hyperparam_config = hyperparam_config
        self.use_wandb = use_wandb
        self.fit_on_init = fit_on_init
        self.model_name_add = model_name_add

        # Define loss functions
        self.cat_loss_fn = nn.BCEWithLogitsLoss()
        #self.cat_loss_fn = nn.CrossEntropyLoss() # TODO: understand this mores
        self.regr_loss_fn = nn.MSELoss()

        # Convert hyperparameter config to simpler parameter dictionary
        self.params = self.get_parameters()

        # Augment data if desired (using random sampling or SMOTE)
        self.augment_data()
        # Turn dataframe into Dataset
        #print(self.target)
        self.ds = DFDataset(dataframe=self.df, inputs=self.inputs, target=self.target)
        # Create model. If using k-fold, we'll remember to zero its weights, etc.
        # to avoid folds influencing each other!
        self.model = self.create_model()

        # Just fit straight away, basically treating this initialiser like any
        # other function.
        self.datetime = None  # Just make it obvious this is an attribute
        if fit_on_init:
            self.fit()

            # added by JC
            # save the results into a csv file
            metric_fpath = self.fit_folder / "tv_metrics.csv"

            self.out_df = { "EPOCH"  : np.arange(self.epochs) + 1, 
                    "TR_LOSS"   : self.train_loss,
                    "VL_LOSS"   : self.val_loss,
                    "VL_F1"     : self.val_f1, 
                    "VL_ACC"    : self.val_acc
                }
            self.out_df = pd.DataFrame(self.out_df)
            self.out_df.to_csv(path_or_buf = metric_fpath, index = False, float_format="%g")
            # end add by JC

    def __getattr__(self, key):
        """
        Allows us to access any hyperparameter stored in `self.params` simply by
        accessing that attribute from this object. For example, if you want the 
        `use_SMOTE` parameter, you can simply do:
        >>> self.use_SMOTE

        Only caveat is you can't have an instance attribute with the same name, 
        but that shouldn't be an issue.
        
        This might be a little unorthodox, but I think it's neat and saves a lot
        of typing (no more `self.params["use_SMOTE"]`!).

        Raises AttributeError if `key` not found in `self.params`.
        """
        if key in self.params:
            return self.params[key]

        raise AttributeError(
            f"'{key}' not found in instance attributes nor self.params!"
        )

    def get_parameters(self) -> dict:
        """
        Convert hyperparameter config into much simpler single-level dictionary
        of parameter keys and their values.
        """
        # Get parameters
        if self.use_wandb:
            wandb.init(config=self.hyperparam_config)
            params = wandb.config
        else:
            # Since hyperparam_config contains some top-level metadata, and we just need
            # the actual parameters and their values.
            params, _ = convert_hyperparam_config_to_values(self.hyperparam_config)
        return params

    def augment_data(self) -> None:
        """
        "Augment" data using random sampling or SMOTE, whichever is specified in
        hyperparameters.

        NOTE: Don't try to use both methods to augment data. They were not 
        designed with that in mind. Only random sampling will occur if you try 
        both.

        NOTE: Modifies `self.df`, will not return a copy.
        """
        if self.use_random_sampling:
            self.augment_random_sampling()
        elif self.use_SMOTE:
            self.augment_SMOTE()

    def augment_random_sampling(self) -> None:
        """
        Augment DataFrame using random sampling, with relative weights for each 
        target category as specified in hyperparameters.

        NOTE: Modifies `self.df`, will not return a copy.
        """
        print(f"Using random sampling (weights = {self.sampling_weights})")
        df = self.df  # Just for shorter reference

        # We can take advantage of pandas DataFrame.sample() method to oversample
        # some categories and undersample others! We just need to create a Series
        # of relative weights, and bam!
        norm_weights = pd.Series(index=df.index)

        # We need to normalise weights to make it so each category's inputs sum
        # to their desired probability!
        for i, weight in enumerate(self.sampling_weights):  # Assume categories start at 0, 1, ...
            # Ensure that the weights of each individual point in this category
            # *add up* to the desired smapling probability.
            cat_indices = df.MW_CAT == i
            n_cat = cat_indices.sum()
            norm_weight = weight / n_cat
            norm_weights[cat_indices] = norm_weight

        # Now use those weights to create the new distributed df
        self.df = df.sample(frac=1, replace=True, weights=norm_weights)
        self.find_and_reindex_duplicate_data()

    def find_and_reindex_duplicate_data(self) -> None:
        """
        Finds duplicate rows in dataframe (i.e. from df.sample()), and 
        re-indexes them to ensure unique indices. Also sets their "SYNTHETIC"
        values to True.
        
        Intended to be used after augmenting using random sampling, which leads
        to these duplicate indices.
        """
        df = self.df

        # Get indices that are duplicates, set their SYNTHETIC, 
        synthetic_indices = df.index.duplicated()
        df.loc[synthetic_indices, "SYNTHETIC"] = True

        # Generate new range of indices for these duplicates
        max_index = max(df.index) # Easy way to find new lowest unique index
        n_synthetic = synthetic_indices.sum()
        new_synthetic_indices = np.arange(start=max_index+1, stop=max_index+n_synthetic+1)
        
        # Get original index, then replace all synthetics with their new, and apply
        # back to original dataframe
        index = df.index.to_numpy()
        index[synthetic_indices] = new_synthetic_indices
        df.index = index
        self.df = df

    def augment_SMOTE(self) -> None:
        """
        Augment DataFrame using SMOTE, with relative weights for each target 
        category as described in hyperparameters.
        """
        print("Using SMOTE")
        df = self.df  # Just for shorter reference

        # Figure out how many samples we want from each target class
        weights = np.array(self.sampling_weights)
        weights /= weights.sum() # Normalise so they sum to one
        # Assume that we want same length of DataFrame (TODO: Make desired length
        # a hyperparameter. Same goes for random sampling!). NOTE: This won't guarantee
        # exact same length as before (rounding!), but close enough.
        sampling_strategy = {
            target_cat: int(weight * len(df)) for target_cat, weight in enumerate(weights)
        }

        # WARN: Bit shaky for me to add target variable (e.g. MW) to SMOTE, 
        # just a temporary workaround!
        X = df[[*self.inputs, self.target, "REGION"]]
        y = df["MW_CAT"].to_numpy()
        sm = SMOTENC(
            categorical_features=[len(X.columns) - 1], # REGION (last input) is categorical!
            random_state=42,
            sampling_strategy=sampling_strategy,
        )
        X_res, y_res = sm.fit_resample(X, y)

        # This SMOTE resets all the indices in the X dataframe, but it's nice
        # because all the synthetic data is just appended to the end of the
        # dataframe. I.e., if you start with 5000 rows, and synthesis another
        # 5000, you can easily distinguish between the real and fake data by
        # just splitting the dataframe!
        is_synthetic = np.zeros(shape=len(X_res), dtype=bool)
        n_synthesised = len(X_res) - len(X)
        is_synthetic[n_synthesised:] = True
        X_res["SYNTHETIC"] = is_synthetic
        X_res["MW_CAT"] = y_res
        self.df = X_res

    def create_model(self) -> EarthquakeMTL:
        """
        Create and return EarthquakeMTL model based on specified 
        hyperparameters. Will move model to `self.DEVICE` for you.
        """
        df = self.df  # Just for shorter reference

        # for the hidden layers
        try: 
            hidden_layers = self.hidden_layers
        except: # only happens when doing sweeps
            hidden_layers = [self.hidden_layers1, self.hidden_layers2]

        # Define the model
        kernel_size = self.kernel_size
        model = EarthquakeMTL(
            n_cont=kernel_size[0] * kernel_size[1] * len(self.inputs),
            n_regions=len(df["REGION"].unique()),
            n_out=len(self.mw_cats) - 1,
            hidden=hidden_layers,
            emb=self.embeddings,
            do=self.dropout,
            bn=self.batch_normalisation,
            activ=self.activation_function,
            categorical_output=self.categorical_output,
            regression_output=self.regression_output
        )

        model = model.to(self.DEVICE)
        return model

    def get_k_fold_val_ids(fself) -> List[Index]:
        """
        Returns a list of pandas Index objects, with each Index to be used as
        the validation fold indices for a fold.

        This was needed since the scikit K-fold uses positional indices (i.e.,
        0..length of dataset - 1), as opposed to the preferred label indices (
        which could be anything, especially since some rows are droppped during
        preprocessing). Although the scikit method could work, it was awkward since
        we use label-based indices everywhere else.

        NOTE: This completely ignores the possibility of synthetic data, so it's
        best to drop synthetic data from the validation ids when we're actually
        going fold by fold.
        """
        df = self.df.copy()  # Just in case you don't want to shuffle the original
        df = df.sample(frac=1)  # Shuffle dataframe before splitting

        fold_val_ids = []  # Array of Index objects (basically array of arrays)
        fold_size = int(len(df) / self.k_folds)  # Size of each k-folds

        # Chunk up dataframe,
        for fold in range(self.k_folds):
            start = fold * fold_size

            # Last fold needs to be treated specially, since it may have more
            # or less rows based on how the chunk size divided the dataset's length
            if fold == self.k_folds - 1:
                val_ids = df.iloc[start:].index  # Just sample remaining rows
            else:
                val_ids = df.iloc[start : start + fold_size].index
            fold_val_ids.append(val_ids)

        return fold_val_ids

    def fit(self) -> dict:
        """
        The meat of this class. This will do all the actual training and 
        validation. Will use k-fold cross validation if specified. Returns
        validation metrics as a dict.

        If not using k-fold, will return the "best_f1" score and "best_accuracy"
        for the single validation loop. If using k-fold, then will return single
        "best_f1" and "best_accuracy" for all folds (i.e. does not return 
        fold-by-fold breakdown), and will also return "ave_f1" and "ave_accuracy",
        which are the averages of each fold's best metrics. These latter metrics
        should be more reliable.
        """
        self.datetime = datetime.now().isoformat()

        df = self.df
        ds = self.ds

        if self.use_k_fold:
            fold_metrics = []

            fold_val_ids = self.get_k_fold_val_ids()
            for fold, val_ids in enumerate(fold_val_ids):
                print(f"FOLD {fold + 1}/{self.k_folds}")
                print("--------------------------------")
                train_ids = df.drop(index=val_ids).index

                # Keep track of this in case we need to move some validation indices
                # down below
                n_val = len(val_ids)

                # We want no synthetic data in validation! Note that we make sure
                # to keep the training/validation split the same by making a
                # one-to-one trade between validation and training datasets.
                val_df = df.loc[val_ids]
                synthetic_val_ids = (val_df[val_df["SYNTHETIC"] == True]).index
                val_ids = [i for i in val_ids if i not in synthetic_val_ids]

                if len(synthetic_val_ids) > 0:
                    print(
                        f"Dropped {len(synthetic_val_ids)} rows from validation since they were synthetic ({len(val_ids)} remaining)"
                    )

                fold_best = self.fit_aux(
                    train_ids=train_ids,
                    val_ids=val_ids,
                    fold=fold
                )
                fold_metrics.append(fold_best)
                print(
                    f"Fold best     -    F1-score: {fold_best['best_f1']*100:.3f};       accuracy: {fold_best['best_accuracy']*100:.3f}%"
                )

            # Summarise all folds now. NOTE: Best f1 and best accuracy could have
            # occurred in different folds, and even if same fold could have
            # occurred at different epochs! Average really seems more useful.
            ave_f1 = np.mean([fold["best_f1"] for fold in fold_metrics])
            ave_accuracy = np.mean([fold["best_accuracy"] for fold in fold_metrics])
            best_f1 = max([fold["best_f1"] for fold in fold_metrics])
            best_accuracy = max([fold["best_accuracy"] for fold in fold_metrics])
            print(
                f"Fold averages -    F1-score: {ave_f1*100:.3f};       accuracy: {ave_accuracy*100:.3f}%"
            )
            print(
                f"Fold bests    -    F1-score: {best_f1*100:.3f};       accuracy: {best_accuracy*100:.3f}%"
            )
            return {
                "ave_f1": ave_f1,
                "ave_accuracy": ave_accuracy,
                "best_f1": max([fold["best_f1"] for fold in fold_metrics]),
                "best_accuracy": max([fold["best_f1"] for fold in fold_metrics]),
            }
        else:
            # Split dataframe into its initial/real data and any synthesised data
            real_indices = df[df["SYNTHETIC"] == False].index

            # Get required number of validation points from real data only!
            n_val = int(len(ds) * self.VALIDATION_FRAC)
            val_ids = rng.choice(real_indices, size=n_val, replace=False)
            train_ids = df.drop(index=val_ids).index

            self.val_ids = val_ids
            return self.fit_aux(train_ids=train_ids, val_ids=val_ids)

    def fit_aux(self, train_ids, val_ids, fold=None) -> dict:
        """
        An auxillary function to reset the model, create the optimiser, scheduler
        and run the training and validation.

        This is useful as its own function since we have optional k-fold
        cross-validation, and the only thing that really changes between k-fold and
        no k-fold is the Dataloaders you use for training and validation.

        Returns a dictionary of the best accuracies achieved.
        """
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Convert Datasets into DataLoaders
        train_loader = torch.utils.data.DataLoader(
            self.ds, batch_size=self.batch_size, sampler=train_subsampler
        )
        val_loader = torch.utils.data.DataLoader(
            self.ds, batch_size=self.batch_size, sampler=val_subsampler
        )

        self.val_loader = val_loader

        self.model.apply(reset_weights)

        # Define the optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Define the scheduler to adjust learning rate across epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            steps_per_epoch=math.ceil(len(train_loader)),
            epochs=self.epochs,
        )

        # Record metrics across all epochs
        all_accuracies = np.zeros(self.epochs)
        all_f1s = np.zeros(self.epochs)

        # added by JC to monitor progress
        self.train_loss = np.zeros(self.epochs)
        self.val_loss = np.zeros(self.epochs)
        self.val_f1 = np.zeros(self.epochs)
        self.val_acc = np.zeros(self.epochs)
        # end add
        # class_accuracy = np.zeros((params["epochs"], num_cats))

        curr_vloss = 1000
        for epoch in range(self.epochs):
            # Train model, find training loss
            train_metrics = self.train_loop(
                train_loader=train_loader, optimizer=optimizer, scheduler=scheduler
            )
            train_loss = train_metrics["comb_loss"]

            # added by JC
            self.train_loss[epoch] = train_metrics["comb_loss"]
    
            self.train_metrics = train_metrics
            # end add
            # print(
            #     f"Train loss         : {train_loss:>7f}  [{epoch+1:>5d}/{self.epochs:>5d}]"
            # )

            # Validate model, find  F1 score and accuracy
            validation_metrics = self.validation_loop(val_loader=val_loader)
            self.validation_metrics = validation_metrics
            
            f1 = validation_metrics["f1"]
            accuracy = validation_metrics["accuracy"]

            # added by JC
            self.val_loss[epoch] = validation_metrics["comb_loss"]
            self.val_acc[epoch] = validation_metrics["accuracy"]
            self.val_f1[epoch] = validation_metrics["f1"]
            # end add

            #print(f"Validation F1-score: {f1:>7f}  [{epoch+1:>5d}/{self.epochs:>5d}]")
            #print(
            #    f"Validation accuracy: {accuracy:>7f}  [{epoch+1:>5d}/{self.epochs:>5d}]"
            #)

            # Store validation results for later
            all_accuracies[epoch] = accuracy
            all_f1s[epoch] = f1
            # for i in range(num_cats):
            #     # class_accuracy[i][epoch] = valid_loss["class_acc"][i]
            #     class_accuracy[epoch][i] = valid_loss["class_acc"][i]

            # Save model 
            # save model during wandb sweep - but only save if validation loss improved
            add_fname = self.params["exclude_file"].removesuffix(".csv")
            add_fname = add_fname + "_" + self.model_name_add 
            fit_folder = default.MODELS_DIRECTORY / (add_fname + "_" + self.datetime)
            if fold is not None:
                fit_folder /= f"fold_{fold}"
            fit_folder.mkdir(parents=True, exist_ok=True) # Make folder(s) iff required
            model_filepath = fit_folder / f"epoch-{epoch}.pt"
            self.fit_folder = fit_folder # used later

            # only save model when validation loss decreases
            if validation_metrics["comb_loss"] < curr_vloss: 
                best_model_fpath = model_filepath
                torch.save(self.model.state_dict(), model_filepath)
                curr_vloss = validation_metrics["comb_loss"] 

            # Summarising all metrics we're interested in
            metrics = {
                "train_loss": train_loss,
                "val_loss": validation_metrics["comb_loss"],
                "total_accuracy": accuracy,
                "f1": f1
                # "class_acc": valid_loss["class_acc"]
            }

            # Print results for this epoch
            # pprint(metrics)
            if self.use_wandb:
                wandb.log(metrics)

        self.fit_folder = fit_folder
        #best_metrics = {"best_accuracy": all_accuracies.max(), "best_f1": all_f1s.max()}
        best_metrics = {"best_val_loss": self.val_loss.min()}
        if self.use_wandb:
            wandb.log(best_metrics)

        return best_metrics

    def train_loop(self, train_loader, optimizer, scheduler) -> dict:
        """
        Simple wrapper around actual `train_loop()` from `train.py` just so
        we don't have to pass arguments which are already stored in this object!
        """
        return train_loop(
            model=self.model,
            train_loader=train_loader,
            cat_loss_fn=self.cat_loss_fn,
            regr_loss_fn=self.regr_loss_fn,
            optimizer=optimizer,
            device=self.DEVICE,
            scheduler=scheduler,
            cat_scaling=self.cat_scaling,
            regr_scaling=self.regr_scaling,
        )

    def validation_loop(self, val_loader) -> dict:
        """
        Simple wrapper around actual `validation_loop()` from `validate.py` just so
        we don't have to pass arguments which are already stored in this object!
        """
        return validation_loop(
            model=self.model,
            val_loader=val_loader,
            cat_loss_fn=self.cat_loss_fn,
            regr_loss_fn=self.regr_loss_fn,
            device=self.DEVICE,
            cat_scaling=self.cat_scaling,
            regr_scaling=self.regr_scaling,
        )