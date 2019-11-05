import torch
import numpy as np

from pltools.data.dataset import BaseCacheDataset


class LoadDummySample:
    def __init__(self, keys=('data', 'label'), sizes=((3, 128, 128), (3,)),
                 **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.sizes = sizes

    def __call__(self, path, *args, **kwargs):
        data = {_k: np.random.rand(*_s)
                for _k, _s in zip(self.keys, self.sizes)}
        data['id'] = f'sample{path}'
        return data


class DummyDataset(BaseCacheDataset):
    def __init__(self, num_samples=10, load_fn=LoadDummySample(),
                 **load_kwargs):
        super().__init__(list(range(num_samples)), load_fn, **load_kwargs)


class Config:
    def __init__(self, **kwargs):
        super().__init__()
        for key, item in kwargs.items():
            setattr(self, key, item)


class DummyModel(torch.nn.Module):
    def __init__(self):
        self.call_count = -1
        self.called = False
        super().__init__()

    def forward(self, *args, **kwargs):
        self.call_count += 1
        self.called = True
        return self.call_count
