"""
A fast and simple data management library for machine learning
"""

from typing import Dict, Callable, Iterable, Self
from functools import reduce, cache
import numpy as np
from numpy.typing import NDArray


class DatasetDict:
    """
    Store and manage split datasets.
    """
    def __init__(self, data: Dict[str, NDArray]):
        """
        Input data is assumed to follow the format `X, Y key -> numpy array`.
        """
        self.__data = data
        length = 0
        for k, v in data.items():
            if length == 0:
                length = len(v)
            elif length != len(v):
                raise AttributeError(f"Data should be composed of equal length arrays, column {k} has length {len(v)} should be {length}")
        self.length = length

    def select(self, idx: int | Iterable[int | bool]):
        """
        Return a new DatasetDict that only contains the samples at the indices specified.

        Arguments:
        - idx: Index or indices of the samples to take from the data
        """
        return DatasetDict({k: v[idx] for k, v in self.__data.items()})

    def __getitem__(self, i: str) -> NDArray:
        return self.__data[i]
    
    def __len__(self) -> int:
        return len(self.__data['X'])
    
    def __str__(self) -> str:
        return str(self.__data)
    
    def short_details(self) -> str:
        "Give shortened details on the structure of the data."
        details = "{"
        for k, v in self.__data.items():
            details += f"{k}: type {v.dtype}, shape {v.shape}, range [{v.min()}, {v.max()}], "
        details = details[:-2] + "}"
        return details


class Dataset:
    """
    Store and manage a whole dataset.
    """
    def __init__(self, data: Dict[str, Dict[str, NDArray] | DatasetDict]):
        """
        Input data when creating a dataset is assumed to follow the format of `train/test/validation/etc. key -> X, Y keys -> numpy array`.
        Data is always assumed to have at least a train key with the corresponding structure underneath.
        """
        if np.all([isinstance(v, DatasetDict) for v in data.values()]):
            self.__data = data
        else:
            self.__data = {k: DatasetDict(v) for k, v in data.items()}
    
    def __getitem__(self, i: str) -> DatasetDict:
        return self.__data[i]

    def __str__(self) -> str:
        string = "{\n"
        for k, v in self.__data.items():
            string += f"\t{k}: {v.short_details()}\n"
        string += "}"
        return string
    
    def keys(self) -> Iterable[str]:
        """
        Get the top level keys of the dataset, i.e., the split of the dataset.
        """
        return self.__data.keys()

    def select(self, idx_dict: Dict[str, int | Iterable[int | bool]]):
        """
        Return a subdataset which includes only the data at the specified indices.

        Arguments:
        - idx_dict: A dictionary with the format of `train/test/validation/etc. key -> numpy array of indices`
        """
        return Dataset({k: self.__data[k].select(idx) for k, idx in idx_dict.items()})

    @property
    @cache
    def nclasses(self):
        return len(np.unique(reduce(np.union1d, [np.unique(d['Y']) for d in self.__data.values()])))

    @property
    @cache
    def input_shape(self):
        return self.__data['train']['X'][0].shape