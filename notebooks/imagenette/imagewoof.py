# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
NOTEBOOK = 0

# %%
from omegaconf import OmegaConf
cfg = OmegaConf.load(r'/home/michael/dev/phoenix/PytorchLightningTools/notebooks/imagenette/config.yaml')

# %%
import numpy as np
from skimage.transform import resize

SIZE = (3, 128, 128)


class PytorchDatasetWrapper():
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        sample = self.dataset[index]
        data = np.array(sample[0]).astype(np.float32).transpose(2, 0, 1)
        return {'data': resize(data, SIZE), 'label': sample[1], 'id': f'sample{index}'}


# %%
import pathlib

PATH = pathlib.Path(cfg.paths.data)
train_path = PATH / 'train'
val_path = PATH / 'val'

# %%
import torch
import numpy as np
import random
from torchvision.datasets import ImageFolder
from pltools.data import ToTensor
from data_loading.experimental import DataLoader
from data_loading import numpy_collate

from batchgenerators.transforms import ZeroMeanUnitVarianceTransform, Compose, MirrorTransform


pre_transforms = [ZeroMeanUnitVarianceTransform()]
train_transforms = [MirrorTransform(axes=(0, 1))]
post_transforms = [ToTensor(
    keys=('data', 'label'),
    dtypes=(('data', torch.float32), ('label', torch.int64)))]


def init_fn(worker_id):
    seed = torch.utils.data._utils.worker._worker_info.seed
    import numpy as np
    seed32 = np.array(seed).astype(np.uint32)
    np.random.seed(seed32)


train_dset = PytorchDatasetWrapper(ImageFolder(train_path))
train_transforms = Compose(pre_transforms + train_transforms + post_transforms)
train_data = DataLoader(train_dset,
                        shuffle=True, batch_size=32, num_workers=4,
                        batch_transforms=train_transforms, collate_fn=numpy_collate,
                        pin_memory=False,
                        worker_init_fn=init_fn)

val_dset = PytorchDatasetWrapper(ImageFolder(val_path))
val_transforms = Compose(pre_transforms + post_transforms)
val_data = DataLoader(val_dset,
                      shuffle=False, batch_size=32, num_workers=4,
                      batch_transforms=val_transforms, collate_fn=numpy_collate,
                      pin_memory=False,
                      worker_init_fn=init_fn)

# %%

if NOTEBOOK:
    import matplotlib.pyplot as plt

    sample = train_dset[32]['data']
    sample -= sample.min()
    sample = sample / sample.max()
    plt.imshow(sample.transpose(1, 2, 0))

    # # %%
    # data_iter = iter(train_data)

    # # %%
    # from torchvision.utils import make_grid
    # import matplotlib.pyplot as plt

    # batch = next(data_iter)
    # print(batch['data'].shape)
    # grid = make_grid(batch['data']).cpu().numpy()
    # grid -= grid.min()
    # grid = grid / grid.max()
    # plt.imshow(grid.transpose((1, 2, 0)))

# %%


# %%
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from pltools.train.module import Module
from collections import defaultdict


class Classifier(Module):
    def __init__(
            self,
            config,
            model=resnet18(
                pretrained=False,
                num_classes=10)):
        super().__init__(config, model=model)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x = batch["data"]
        y = batch["label"]
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        tensorboard_logs = {'train_loss': loss}
        if batch_nb in [0]: # list(range(1)):
            normed_batch = x.clone().detach()
            normed_batch -= normed_batch.min()
            normed_batch = normed_batch / normed_batch.max()
            self.logger.experiment.add_images('first_train_batch', normed_batch)
            self.logger.experiment.add_text('first_train_ids', f'{batch["id"]}')

        return {'loss': loss,
                'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch["data"]
        y = batch["label"]
        y_hat = self.forward(x)

        val_loss = F.cross_entropy(y_hat, y, reduction='mean')

        # calculate acc
        with torch.no_grad():
            labels_hat = torch.argmax(y_hat.detach(), dim=1)
            val_acc = torch.sum(y == labels_hat) / (len(y) * 1.0)
        return {'val_loss': val_loss, 'val_acc': val_acc}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_val = defaultdict(list)
        for listitem in outputs:
            for key, item in listitem.items():
                avg_val[key].append(item)

        avg_val = {key: torch.stack(item).mean()
                   for key, item in avg_val.items()}

        print("-----------------------ValEnd------------------------")
        return {'avg_val_loss': avg_val['val_loss'], 'log': avg_val}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams["training"]["lr"])


# %%
# Create module and save data into module
module = Classifier(cfg)
module.train_data = train_data
module.val_data = val_data

# %%
from pltools.train import lr_find, plot_lr_curve


# if NOTEBOOK:
#     lrs, losses = lr_find(module, gpu_id=0)
#     plot_lr_curve(lrs, losses)

# %%
# plot_lr_curve(lrs[10:-10], losses[10:-10])


# %%
import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
import os

logger = TestTubeLogger(save_dir=os.getcwd(), name=cfg["exp_name"])
trainer = pl.Trainer(gpus=[0], amp_level='O1', use_amp=False,
                     fast_dev_run=False, overfit_pct=0.0,
                     early_stop_callback=None,
                     max_nb_epochs=cfg["training"]["epochs"],
                     logger=logger)
trainer.fit(module)


# %%
