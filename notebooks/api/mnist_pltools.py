# %%
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import os

dset = MNIST(os.getcwd(), train=True, download=True,
             transform=None)
# %%
import torch
import numpy as np
from pltools.data.transformer import Transformer, ToTensor

from batchgenerators.transforms import ZeroMeanUnitVarianceTransform, \
    MirrorTransform


class MNIST_Wrapper():
    def __init__(self, dset):
        self.dset = dset

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        sample = self.dset[index]
        return {'data': np.array(sample[0])[None].astype(np.float32),
                'label': sample[1]}


dset_dict = MNIST_Wrapper(dset)
dset_transformer = Transformer(dset_dict,
                               [MirrorTransform((0,)),
                                ToTensor(dtypes=(
                                    ('data', torch.float32),
                                    ('label', torch.int64)))])
# %%
dset_dict[1]['data'].shape
# %%
dset_transformer[1]['data'].shape

# %%
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class CoolSystem(pl.LightningModule):
    def __init__(self):
        super(CoolSystem, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x = batch["data"]
        y = batch["label"]
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch["data"]
        y = batch["label"]
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(dset_transformer, batch_size=32, num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(dset_transformer, batch_size=32, num_workers=4)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(dset_transformer, batch_size=32, num_workers=4)


# %%
from pytorch_lightning import Trainer

model = CoolSystem()

# most basic trainer, uses good defaults
trainer = Trainer()
trainer.fit(model)


# %%
