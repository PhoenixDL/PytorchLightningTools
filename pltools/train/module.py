from __future__ import annotations


import typing
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pltools.config import Config

transform_type = typing.Iterable[typing.Callable]


class Module(pl.LightningModule):
    def __init__(self,
                 hparams: Config,
                 model: torch.nn.Module,
                 train_data: DataLoader = None,
                 val_data: DataLoader = None,
                 test_data: DataLoader = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self._initial_optimizers = None
        self._initial_forward = None

    def forward(self, data: torch.Tensor, *args, **kwargs):
        return self.model(data, *args, **kwargs)

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        if self.train_data is None:
            return super().train_dataloader()
        return self.train_data

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        if self.val_data is None:
            return super().val_dataloader()
        return self.val_data

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        if self.test_data is None:
            return super().test_dataloader()
        return self.test_data

    @property
    def val_data(self):
        return self._get_internal_dataloader("val")

    @val_data.setter
    def val_data(self, loader):
        self._set_internal_dataloader("val", loader)

    @property
    def test_data(self):
        return self._get_internal_dataloader("test")

    @test_data.setter
    def test_data(self, loader):
        self._set_internal_dataloader("test", loader)

    def _get_internal_dataloader(self, name):
        return getattr(self, f'_{name}_loader')

    def _set_internal_dataloader(self, name, loader):
        setattr(self, f'_{name}_loader', loader)
        if (loader is not None and
                hasattr(self, f'_lazy_{name}_dataloader')):
            delattr(self, f'_lazy_{name}_dataloader')

    def enable_tta(self,
                   trafos: transform_type = (),
                   inverse_trafos: transform_type = None,
                   tta_reduce: typing.Callable = None,
                   ) -> None:
        self._initial_forward = self.forward
        self.forward = tta_wrapper(self.forward,
                                   trafos=trafos,
                                   inverse_trafos=inverse_trafos,
                                   tta_reduce=tta_reduce,
                                   )

    def disable_tta(self) -> bool:
        if self._initial_forward is not None:
            self.forward = self._initial_forward
            self._initial_forward = None
            return True
        else:
            return False


def tta_wrapper(func: typing.Callable,
                trafos: typing.Iterable[typing.Callable] = (),
                inverse_trafos: typing.Iterable[typing.Callable] = None,
                tta_reduce: typing.Callable = None,
                ) -> typing.Callable:
    _trafo = (None, *trafos)
    _inverse_trafos = (None, *inverse_trafos)

    def tta_forward(data: torch.Tensor, *args,
                    **kwargs) -> typing.Any:
        tta_preds = []
        for idx, t in enumerate(_trafo):
            tta_data = t(data) if t is not None else data

            tta_pred = func(tta_data, *args, **kwargs)

            if (_inverse_trafos is not None and
                    _inverse_trafos[idx] is not None):
                tta_pred = _inverse_trafos[idx](tta_pred)

            tta_preds.append(tta_pred)

        if tta_reduce is not None:
            tta_preds = tta_reduce(tta_preds)
        return tta_preds
    return tta_forward
