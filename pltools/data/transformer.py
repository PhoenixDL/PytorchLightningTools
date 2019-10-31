import typing
import numpy as np
import torch
from .dataset import AbstractDataset

from batchgenerators.transforms import Compose


class Transformer(object):
    def __init__(self, dataset: AbstractDataset,
                 transforms: typing.Iterable,
                 compose: typing.Callable = Compose,
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.compose = compose
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, attr: str) -> typing.Any:
        # NOTE do note use hasattr, it goes into infinite recursion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        return getattr(self._experiment, attr)

    def __getitem__(self, index: int) -> typing.Any:
        data = self.dataset[index]
        batch = {key: np.asarray(item)[None] for key, item in data.items()}
        batch = self._transforms(**batch)
        batch = {key: item[0] for key, item in batch.items()}
        return batch

    @property
    def transforms(self) -> typing.Iterable:
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: typing.Iterable):
        self._transforms = self.compose(transforms)


class ToTensor(object):
    def __init__(self, keys: typing.Iterable = None,
                 dtypes: typing.Iterable[tuple] = None):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, **data) -> dict:
        _keys = self._resolve_keys(data)

        torch_data = {}
        for key, item in data.items():
            if key in _keys:
                torch_data[key] = torch.from_numpy(item)
            else:
                torch_data[key] = item

        if self.dtypes is not None:
            for _dtype in self.dtypes:
                torch_data[_dtype[0]] = torch_data[_dtype[0]].to(
                    dtype=_dtype[1])

        return torch_data

    def _resolve_keys(self, data):
        if self.keys is None:
            _keys = data.keys()
        else:
            _keys = self.keys
        return _keys
