import json
import pickle

import torch
import yaml

import default
from preprocessor import Preprocessor

ZERO_TENSOR = torch.zeros(1)  # Just useful as a placeholder for emtpy losses later


def load_data(
    data_folder,
    exclude_file, # added by JC
    cats,
    kernel_size,
    rand_seed = None,
    skip_drop_na = False, 
    rd_exclude = False, # added by JC 
    use_cache=True,
    pickle_fp="preprocessed.pickle",
    **kwargs
) -> Preprocessor:
    """
    Returns *preprocessed* data from python pickle file if available (and if
    `cached` is True). Otherwise, loads data from original files, preprocesses
    that data, pickles that object for later retrieval and then returns.

    Any additional `kwargs` will be passed into the `Preprocessor` initialiser
    if `use_cache` is `False` or if the cache-read failed. I'm too lazy to
    perfectly match this functions input parameters to those for `Preprocessor`.
    Sue me.

    This is to basically speed up execution of the script. Unless you change
    something about the preprocessing, you only really need to do it once and
    save the results, so you can get to the actual training of the model much
    more quickly!
    """
    pickle_file = data_folder / pickle_fp

    if use_cache:
        try:
            with open(pickle_file, "rb") as f:
                preprocessed_data = pickle.load(f)
                return preprocessed_data
        except (FileNotFoundError, EOFError):
            print("Didn't find cached Preprocessor data. Preprocessing instead.")

    # Either user did not want cache, or cache was not found/was empty
    preprocessed_data = Preprocessor(
        data_folder=data_folder, exclude_file = exclude_file, quake_cat_list=cats, rand_seed = rand_seed, skip_drop_na = skip_drop_na, rd_exclude = rd_exclude, kernel_size=kernel_size, **kwargs
    )

    # Now pickle the new result and return
    with open(pickle_file, "wb") as f:
        pickle.dump(preprocessed_data, f)
    return preprocessed_data


def get_config(key, default):
    """
    Helper function which reads from `config.json` file. This lets us update
    the config in the middle of a run, and have those changes reflected in the
    code! Useful when running this script as a notebook.
    """
    # Read configuration from file
    with open("config.json", "r") as f:
        config = json.load(f)

    return config.get(key, default)


# Reset weights for K-Fold Validation
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def deep_update(mapping, *updating_mappings):
    """
    Copied from pydantic's utils functions. Just didn't want to pip install the
    whole package just for this one function lol.

    Useful function to do nested updates on dicts WITHOUT overwriting
    sub-dictionaries. The default dict.update() is pretty crap for this, I'm
    surprised there is no built-in for this type of deep update.

    Link: https://github.com/samuelcolvin/pydantic/blob/master/pydantic/utils.py
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def convert_hyperparam_config_to_values(hyperparam_config):
    """
    hyperparam_config will look something like this:
        {
            "parameter_set": "probabilities",
            "dataset": "16k_more_vars",
            "parameters": {
                "epochs": {
                    "desc": "Number of epochs you wish to use in each training loop. Note this will be multiplied if you're using k-fold as well!",
                    "value": 20,
                },
                "k_folds": {"desc": "Number of k-fold labels you wish to use.", "value": 5},
                "activation_function": {
                    "desc": "Name of activation function you wish model to use. This string is then parsed somewhere in the model to determine the corresponding python function used.",
                    "value": "leaky.2",
                },
                "hidden_layers": {
                    "desc": "List of numbers of neurons you want in each hidden layer.",
                    "value": [400, 200],
                },
                "sampling_weights": {
                    "desc": "Random sampling probabilities for each MW category, in ascending order of MW magnitudes.",
                    "values": [
                        [0.35, 0.35, 0.3],
                        [0.3, 0.4, 0.3],
                        [0.3, 0.3, 0.4],
                        [0.4, 0.3, 0.3],
                    ],
                },
                ... and so on
            },
            "method": "bayes",
            "metric": {"goal": "minimize", "name": "ave_reg_great_divergence"},
        }

    For the most part, we only really need the "parameters" for the actual
    computations. Further, we only need their corresponding "value" or "values".
    This function returns a dict with just those parameters as keys, and their
    corresponding value/s in the top level of the dict. E.g., if you want the
    list of lists of probabities, just go params["sampling_weights"] (where params is the
    output of this function).
    """
    params = {}
    is_iterable = {}
    for name, param in hyperparam_config["parameters"].items():
        try:
            values = param["values"]
            params[name] = values
            is_iterable[name] = True
        except KeyError:  # Means we only have single value to worry about
            try: 
                value = param["value"]
                params[name] = value
                is_iterable[name] = False
            except: # it has min and max
                print("WARNING: wandb config is being used ... ")
                pass # just let it pass as this case happens when doing sweeps

    return params, is_iterable

def get_full_hyperparam_config(config_override_file=None):
    """
    Gets default hyperparameter config and overwrites any parameters set in
    specified file.
    """
    print("--- Reading from default.yml to build up initial hyperparameters dictionary...")
    with open(default.WANDB_PARAMETERS_DIRECTORY / "default.yml", "r") as f:
        # safe_load ensures we only load basic python types (ints, floats, lists),
        # which should be all we need. If you ever need to store functions though,
        # see the pyyaml docs.
        hyperparam_config = yaml.safe_load(f)

    # Now override with any values from another yaml file
    print(
        "--- Checking to see if we want to override default hyperparameters with new file..."
    )
    if config_override_file is not None:
        print(
            f"--- Override file found! Updating with hyperparameters from {config_override_file}..."
        )
        with open(default.WANDB_PARAMETERS_DIRECTORY / config_override_file, "r") as f:
            config_override = yaml.safe_load(f)
            hyperparam_config = deep_update(hyperparam_config, config_override)

            # For now, let's assume any value vs. values issues arise only in the
            # level of hyperparam_config["parameters"][<parameter>]
            for d in hyperparam_config["parameters"].values():
                # Delete "value" only if values is also given
                if "values" in d and "value" in d:
                    del d["value"]
            
            # is it correct to do this?
            for d in hyperparam_config["parameters"].values():
                # also delete "value" if "max" and "min" are given
                if (("max" in d) and ("min" in d)) and "value" in d:
                    del d["value"]
            
            # remove hidden_layers key if hidden_layers1 and hidden_layers2 exists
            # hidden_layers1 and hidden_layers2 are used during sweeps 
            if ("hidden_layers1" in hyperparam_config["parameters"].keys() and 
                "hidden_layers2" in hyperparam_config["parameters"].keys()):
                try:
                    del hyperparam_config["parameters"]["hidden_layers"]
                except KeyError:  # just try to delete this if it is present
                    pass  

    return hyperparam_config
