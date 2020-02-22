import abc
import os
import typing
import pathlib

from collections import Iterable
from tqdm import tqdm

from torch.utils.data import Dataset as TorchDset


class Dataset(TorchDset):
    """
    Extension of PyTorch's Datasets by a ``get_subset`` method which returns a
    sub-dataset.

    """

    def get_subset(self, indices):
        """
        Returns a Subset of the current dataset based on given indices
        Parameters
        ----------
        indices : iterable
            valid indices to extract subset from current dataset
        Returns
        -------
        :class:`SubsetDataset`
            the subset
        """

        # extract other important attributes from current dataset
        kwargs = {}

        for key, val in vars(self).items():
            if not (key.startswith("__") and key.endswith("__")):

                if key == "data":
                    continue
                kwargs[key] = val

        kwargs["old_getitem"] = self.__class__.__getitem__
        subset_data = [self[idx] for idx in indices]

        return SubsetDataset(subset_data, **kwargs)


# NOTE: For backward compatibility (should be removed ASAP)
AbstractDataset = Dataset


class SubsetDataset(Dataset):
    """
    A Dataset loading the data, which has been passed
    in it's ``__init__`` by it's ``_sample_fn``
    """

    def __init__(self, data, old_getitem, **kwargs):
        """
        Parameters
        ----------

        data : iterable
            data to load (subset of original data)
        old_getitem : function
            get item method of previous dataset
        **kwargs :
            additional keyword arguments (are set as class attribute)
        """
        super().__init__(None, None)

        self.data = data
        self._old_getitem = old_getitem

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __getitem__(self, index):
        """
        returns single sample corresponding to ``index`` via the old get_item

        Parameters
        ----------
        index : int
            index specifying the data to load

        Returns
        -------
        dict
            dictionary containing a single sample
        """
        return self._old_getitem(self, index)

    def __len__(self):
        """
        returns the length of the dataset

        Returns
        -------
        int
            number of samples
        """
        return len(self.data)


class CacheDataset(Dataset):
    def __init__(self,
                 data_path: typing.Union[typing.Union[pathlib.Path,
                                                      str],
                                         list],
                 load_fn: typing.Callable,
                 mode: str = "append",
                 **load_kwargs):
        """
        Supported modes are: :param:`append` and :param:`extend`
        """
        super().__init__()
        self._load_fn = load_fn
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset(data_path, mode)

    def _make_dataset(self, path: typing.Union[typing.Union[pathlib.Path, str],
                                               list],
                      mode: str) -> typing.List[dict]:
        data = []
        if not isinstance(path, list):
            assert os.path.isdir(path), '%s is not a valid directory' % path
            path = [os.path.join(path, p) for p in os.listdir(path)]

        for p in tqdm(path, unit='samples', desc="Loading samples"):
            self._add_item(data, self._load_fn(p, **self._load_kwargs), mode)
        return data

    @staticmethod
    def _add_item(data: list, item: typing.Any, mode: str):
        _mode = mode.lower()

        if _mode == 'append':
            data.append(item)
        elif _mode == 'extend':
            data.extend(item)
        else:
            raise TypeError(f"Unknown mode detected: {mode} not supported.")

    def __getitem__(self, index: int) -> dict:
        data_dict = self._load_fn(self.data[index], **self._load_kwargs)
        return data_dict


class LazyDataset(Dataset):
    def __init__(self, data_path: typing.Union[str, list],
                 load_fn: typing.Callable,
                 **load_kwargs):
        super().__init__()
        self._load_fn = load_fn
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset(data_path)

    def _make_dataset(self, path: typing.Union[typing.Union[pathlib.Path, str],
                                               list]) -> typing.List[dict]:
        if not isinstance(path, list):
            assert os.path.isdir(path), '%s is not a valid directory' % path
            path = [os.path.join(path, p) for p in os.listdir(path)]
        return path

    def __getitem__(self, index: int) -> dict:
        data_dict = self._load_fn(self.data[index],
                                  **self._load_kwargs)
        return data_dict


class IDManager:
    def __init__(self, id_key: str, cache_ids: bool = True, **kwargs):
        """
        Helper class to add additional functionality to BaseDatasets
        """
        self.id_key = id_key
        self._cached_ids = None

        if cache_ids:
            self.cache_ids()

    def cache_ids(self):
        self._cached_ids = {
            sample[self.id_key]: idx for idx, sample in enumerate(self)}

    def _find_index_iterative(self, id: str) -> int:
        for idx, sample in enumerate(self):
            if sample[self.id_key] == id:
                return idx
        raise KeyError(f"ID {id} not found.")

    def get_sample_by_id(self, id: str) -> dict:
        return self[self.get_index_by_id(id)]

    def get_index_by_id(self, id: str) -> int:
        if self._cached_ids is not None:
            return self._cached_ids[id]
        else:
            return self._find_index_iterative(id)


class CacheDatasetID(CacheDataset, IDManager):
    def __init__(self, data_path, load_fn, id_key, cache_ids=True,
                 **kwargs):
        super().__init__(data_path, load_fn, **kwargs)
        # check if AbstractDataset did not call IDManager with super
        if not hasattr(self, "id_key"):
            IDManager.__init__(self, id_key, cache_ids=cache_ids)


class LazyDatasetID(LazyDataset, IDManager):
    def __init__(self, data_path, load_fn, id_key, cache_ids=True,
                 **kwargs):
        super().__init__(data_path, load_fn, **kwargs)
        # check if AbstractDataset did not call IDManager with super
        if not hasattr(self, "id_key"):
            IDManager.__init__(self, id_key, cache_ids=cache_ids)
