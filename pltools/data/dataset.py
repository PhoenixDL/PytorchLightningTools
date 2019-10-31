import abc
import os
import typing
import pathlib

from collections import Iterable
from tqdm import tqdm


class AbstractDataset:
    """
    Base Class for Dataset
    """

    def __init__(self, data_path: str, load_fn: typing.Callable):
        """
        Parameters
        ----------
        data_path : str
            path to data samples
        load_fn : function
            function to load single sample
        """
        self.data_path = data_path
        self._load_fn = load_fn
        self.data = []

    @abc.abstractmethod
    def _make_dataset(self, path: str):
        """
        Create dataset
        Parameters
        ----------
        path : str
            path to data samples
        Returns
        -------
        list
            data: List of sample paths if lazy; List of samples if not
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        """
        return data with given index (and loads it before if lazy)
        Parameters
        ----------
        index : int
            index of data
        Returns
        -------
        dict
            data
        """
        pass

    def __len__(self):
        """
        Return number of samples
        Returns
        -------
        int
            number of samples
        """
        return len(self.data)

    def __iter__(self):
        """
        Return an iterator for the dataset
        Returns
        -------
        object
            a single sample
        """
        return _DatasetIter(self)

    def get_sample_from_index(self, index):
        """
        Returns the data sample for a given index
        (without any loading if it would be necessary)
        This implements the base case and can be subclassed
        for index mappings.
        The actual loading behaviour (lazy or cached) should be
        implemented in ``__getitem__``
        See Also
        --------
        :method:BaseLazyDataset.__getitem__
        :method:BaseCacheDataset.__getitem__
        Parameters
        ----------
        index : int
            index corresponding to targeted sample
        Returns
        -------
        Any
            sample corresponding to given index
        """

        return self.data[index]

    def get_subset(self, indices):
        """
        Returns a Subset of the current dataset based on given indices
        Parameters
        ----------
        indices : iterable
            valid indices to extract subset from current dataset
        Returns
        -------
        :class:`BlankDataset`
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
        subset_data = [self.get_sample_from_index(idx) for idx in indices]

        return BlankDataset(subset_data, **kwargs)


class _DatasetIter(object):
    """
    Iterator for dataset
    """

    def __init__(self, dset):
        """
        Parameters
        ----------
        dset: :class: `AbstractDataset`
            the dataset which should be iterated
        """
        self._dset = dset
        self._curr_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_index >= len(self._dset):
            raise StopIteration

        sample = self._dset[self._curr_index]
        self._curr_index += 1
        return sample


class BlankDataset(AbstractDataset):
    """
    Blank Dataset loading the data, which has been passed
    in it's ``__init__`` by it's ``_sample_fn``
    """

    def __init__(self, data, old_getitem, **kwargs):
        """
        Parameters
        ----------
        data : iterable
            data to load
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
        returns single sample corresponding to ``index`` via the ``_sample_fn``
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


class ConcatDataset(AbstractDataset):
    def __init__(self, *datasets):
        """
        Concatenate multiple datasets to one

        Parameters
        ----------
        datasets:
            variable number of datasets
        """
        super().__init__(None, None)

        # check if first item in datasets is list and datasets is of length 1
        if (len(datasets) == 1) and isinstance(datasets[0], list):
            datasets = datasets[0]

        self.data = datasets

    def get_sample_from_index(self, index: int) -> typing.Any:
        """
        Returns the data sample for a given index
        (without any loading if it would be necessary)
        This method implements the index mapping of a global index to
        the subindices for each dataset.
        The actual loading behaviour (lazy or cached) should be
        implemented in ``__getitem__``

        See Also
        --------
        :method:AbstractDataset.get_sample_from_index
        :method:BaseLazyDataset.__getitem__
        :method:BaseCacheDataset.__getitem__

        Parameters
        ----------
        index : int
            index corresponding to targeted sample

        Returns
        -------
        Any
            sample corresponding to given index
        """
        curr_max_index = 0
        for dset in self.data:
            prev_max_index = curr_max_index
            curr_max_index += len(dset)

            if prev_max_index <= index < curr_max_index:
                return dset[index - prev_max_index]
            else:
                continue

        raise IndexError("Index %d is out of range for %d items in datasets" %
                         (index, len(self)))

    def __getitem__(self, index: int) -> typing.Any:
        return self.get_sample_from_index(index)

    def __len__(self) -> int:
        return sum([len(dset) for dset in self.data])


class BaseCacheDataset(AbstractDataset):
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
        super().__init__(data_path, load_fn)
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
        data_dict = self.get_sample_from_index(index)
        return data_dict


class BaseLazyDataset(AbstractDataset):
    def __init__(self, data_path: typing.Union[str, list],
                 load_fn: typing.Callable,
                 **load_kwargs):
        super().__init__(data_path, load_fn)
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset(self.data_path)

    def _make_dataset(self, path: typing.Union[typing.Union[pathlib.Path, str],
                                               list]) -> typing.List[dict]:
        if not isinstance(path, list):
            assert os.path.isdir(path), '%s is not a valid directory' % path
            path = [os.path.join(path, p) for p in os.listdir(path)]
        return path

    def __getitem__(self, index: int) -> dict:
        data_dict = self._load_fn(self.get_sample_from_index(index),
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


class CacheDatasetID(BaseCacheDataset, IDManager):
    def __init__(self, data_path, load_fn, id_key, cache_ids=True,
                 **kwargs):
        super().__init__(data_path, load_fn, **kwargs)
        # check if AbstractDataset did not call IDManager with super
        if not hasattr(self, "id_key"):
            IDManager.__init__(self, id_key, cache_ids=cache_ids)


class LazyDatasetID(BaseLazyDataset, IDManager):
    def __init__(self, data_path, load_fn, id_key, cache_ids=True,
                 **kwargs):
        super().__init__(data_path, load_fn, **kwargs)
        # check if AbstractDataset did not call IDManager with super
        if not hasattr(self, "id_key"):
            IDManager.__init__(self, id_key, cache_ids=cache_ids)
