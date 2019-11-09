from __future__ import annotations


import typing
import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl

from pltools.data import Transformer
from omegaconf import DictConfig

transform_type = typing.Iterable[typing.Callable]


class PLTModule(pl.LightningModule):
    def __init__(self,
                 hparams: DictConfig,
                 model: torch.nn.Module,
                 train_transformer: Transformer = None,
                 val_transformer: Transformer = None,
                 test_transformer: Transformer = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        self.model = model
        self.train_transformer = train_transformer
        self.val_transformer = val_transformer
        self.test_transformer = test_transformer
        self._initial_optimizers = None
        self._initial_forward = None

    def forward(self, data: torch.Tensor, *args, **kwargs):
        return self.model(data, *args, **kwargs)

    @pl.data_loader
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_transformer is None:
            return super().train_dataloader()
        kwargs = self.get_dataloading_kwargs("train_dataloader")
        return DataLoader(self.train_transformer, **kwargs)

    @pl.data_loader
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_transformer is None:
            return super().val_dataloader()
        kwargs = self.get_dataloading_kwargs("val_dataloader")
        return DataLoader(self.val_transformer, **kwargs)

    @pl.data_loader
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if self.test_transformer is None:
            return super().test_dataloader()
        kwargs = self.get_dataloading_kwargs("test_dataloader")
        return DataLoader(self.test_transformer, **kwargs)

    def get_dataloading_kwargs(self, name: str):
        if hasattr(self.hparams, name):
            return getattr(self.hparams, name)
        elif hasattr(self.hparams, 'dataloader'):
            return self.hparams.dataloader
        else:
            return {}

    @property
    def val_transformer(self):
        return self._get_transformer("val")

    @val_transformer.setter
    def val_transformer(self, transformer):
        self._set_transformer("val", transformer)

    @property
    def test_transformer(self):
        return self._get_transformer("test")

    @test_transformer.setter
    def test_transformer(self, transformer):
        self._set_transformer("test", transformer)

    def _get_transformer(self, name):
        return getattr(self, f'_{name}_transformer')

    def _set_transformer(self, name, transformer):
        setattr(self, f'_{name}_transformer', transformer)
        if (transformer is not None and
                hasattr(self, f'_lazy_{name}_dataloader')):
            delattr(self, f'_lazy_{name}_dataloader')

    def enable_tta(self,
                   trafos: transform_type = (),
                   inverse_trafos: transform_type = None,
                   tta_reduce: typing.Callable = None,
                   ) -> None:
        self._initial_forward = self.forward
        self.forward = tta_wrapper(self.forward, self,
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
                module: PLTModule,
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
