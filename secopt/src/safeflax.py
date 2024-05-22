"""
Library for saving flax model parameters/variables in the safetensors format.
"""

from typing import Dict, T
import safetensors
import safetensors.flax


def flatten_params(params: Dict[str, Dict | T], key_prefix: str = "", key_separator: str = ":::") -> Dict[str, T]:
    """
    Mainly for internal use. Flattens a PyTree into Dict with string keys and array values.

    Parameters:
    - params: The model parameters/variables as a PyTree
    - key_prefix: The keys in the outer levels for the PyTree during recursion
    - key_separator: Seperator string value to state the different levels for unflattening, choose so it does not clash with
                     the layer names
    """
    flattened_params = {}
    for key, value in params.items():
        key_name = key if key_prefix == "" else f"{key_prefix}{key_separator}{key}"
        if isinstance(value, dict):
            flattened_params.update(flatten_params(value, key_prefix=key_name))
        else:
            flattened_params[key_name] = value
    return flattened_params


def unflatten_params(params: Dict[str, T], key_separator=":::") -> Dict[str, Dict | T]:
    """
    Mainly for internal use. Unflattens a PyTree from a Dict with string keys and array values into the original Flax model's
    structure.

    Parameters:
    - params: The model parameters/variables as a PyTree
    - key_separator: Seperator string value to state the different levels for unflattening, chosen so it does not clash with
                     the layer names
    """
    unflattened_params = {}
    for key, value in params.items():
        subkeys = key.split(key_separator)
        unflattened_params_tmp = unflattened_params
        for subkey in subkeys[:-1]:
            if not unflattened_params_tmp.get(subkey):
                unflattened_params_tmp[subkey] = {}
            unflattened_params_tmp = unflattened_params_tmp[subkey]
        unflattened_params_tmp[subkeys[-1]] = value
    return unflattened_params


def save_file(params: Dict[str, Dict | T], filename: str, key_separator: str = ":::"):
    """
    Save Flax model parameters/variables to a file in the safetensors format.

    Parameters:
    - params: The model parameters/variables as a PyTree
    - filename: Name of the file to save to
    - key_separator: Seperator string value to state the different levels of the parameters dictionary, choose so it does
                     not clash with the layer names
    """
    safetensors.flax.save_file(flatten_params(params, key_separator=key_separator), filename)


def load_file(filename: str, key_separator: str = ":::") -> Dict[str, Dict | T]:
    """
    Load Flax model parameters/variables from a file in the safetensors format.

    Parameters:
    - filename: Name of the file to load from
    - key_separator: Seperator string value to state the different levels of the parameters dictionary, chosen so it does
                     not clash with the layer names
    """
    flat_params = {}
    with safetensors.safe_open(filename, framework="flax") as f:
        for k in f.keys():
            flat_params[k] = f.get_tensor(k)
    return unflatten_params(flat_params, key_separator=key_separator)
