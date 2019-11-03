import typing
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class PLTModule(pl.LightningModule):
    def __init__(self, config, model, train_transformer=None,
                 val_transformer=None, test_transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.model = model
        self.train_transformer = train_transformer
        self.val_transformer = val_transformer
        self.test_transformer = test_transformer

    def forward(self, data: torch.Tensor, *args, **kwargs):
        return self.model(data, *args, **kwargs)

    @pl.data_loader
    def train_dataloader(self):
        if hasattr(self, "train_transformer"):
            kwargs = self.get_dataloading_kwargs("train_dataloader")
            return DataLoader(self.train_transformer, **kwargs)
        else:
            return super().train_dataloader()

    @pl.data_loader
    def val_dataloader(self):
        if hasattr(self, "val_transformer"):
            kwargs = self.get_dataloading_kwargs("val_dataloader")
            return DataLoader(self.val_transformer, **kwargs)
        else:
            return super().val_dataloader()

    @pl.data_loader
    def test_dataloader(self):
        if hasattr(self, "test_transformer"):
            kwargs = self.get_dataloading_kwargs("test_dataloader")
            return DataLoader(self.test_transformer, **kwargs)
        else:
            return super().test_dataloader()

    def get_dataloading_kwargs(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)
        else:
            return self.config.dataloader

    def enable_tta_ensemble(self,
                            trafos: typing.Iterable[typing.Callable] = (),
                            inverse_trafos: typing.Iterable[typing.Callable] = None,
                            tta_reduce: typing.Callable = None,
                            models: typing.Iterable[torch.nn.Module] = None,
                            ensemble_reduce: typing.Callable = None):
        self._initial_forward = self.forward
        self.forward = tta_ensemble_wrapper(self.forward, self,
                                            trafos=trafos,
                                            inverse_trafos=inverse_trafos,
                                            tta_reduce=tta_reduce,
                                            models=models,
                                            ensemble_reduce=ensemble_reduce)

    def disable_tta_ensemble(self, raise_error=False):
        if hasattr(self, "_initial_forward"):
            self.forward = self._initial_forward
        elif not raise_error:
            raise RuntimeError("TTA was not enabled!")
        else:
            pass


def tta_ensemble_wrapper(func, module,
                         trafos: typing.Iterable[typing.Callable] = (),
                         inverse_trafos: typing.Iterable[typing.Callable] = None,
                         tta_reduce: typing.Callable = None,
                         models: typing.Iterable[torch.nn.Module] = None,
                         ensemble_reduce: typing.Callable = None):
    initial_model = module.model

    def tta_ensemable_forward(self, data, *args, **kwargs):
        ensemble_preds = []
        for _model in models:
            module.module = _model
            tta_preds = []
            for idx, _trafo in enumerate(trafos):
                tta_data = _trafo(data) if _trafo is not None else data

                tta_pred = func(tta_data, **kwargs)

                if (inverse_trafos is not None and
                        inverse_trafos[idx] is not None):
                    tta_pred = inverse_trafos[idx](tta_pred)

                tta_preds.append(tta_pred)

                if tta_reduce is not None:
                    tta_preds = tta_reduce(tta_preds)

                ensemble_preds.append(tta_preds)
        if tta_reduce is not None:
            ensemble_preds = ensemble_reduce(ensemble_preds)
    module.model = initial_model
    return tta_ensemable_forward
