import torch
import numpy as np

from pltools.data.dataset import CacheDataset, CacheDatasetID
from pltools.train import Module


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


class DummyDataset(CacheDataset):
    def __init__(self, num_samples=10, load_fn=LoadDummySample(),
                 **load_kwargs):
        super().__init__(list(range(num_samples)), load_fn, **load_kwargs)


class DummyDatasetID(CacheDatasetID):
    def __init__(self, num_samples=10, load_fn=LoadDummySample(),
                 **load_kwargs):
        super().__init__(list(range(num_samples)), load_fn, id_key="id",
                         **load_kwargs)


class Config:
    def __init__(self, **kwargs):
        super().__init__()
        for key, item in kwargs.items():
            setattr(self, key, item)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = -1
        self.called = False
        self.fn = torch.nn.Linear(10, 10)

    def forward(self, *args, **kwargs):
        self.call_count += 1
        self.called = True
        return self.call_count


class DummyModule(Module):
    def __init__(self, length=1000):
        super().__init__(Config(), DummyModel())
        self.length = length

    def training_step(self, *args, **kwargs):
        return {"loss": torch.tensor([1.], requires_grad=True)}

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader([1] * self.length)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0003)
