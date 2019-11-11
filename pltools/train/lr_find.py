import typing
import torch
import tempfile
import pathlib

from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl

IterOptimizer = typing.Iterable[Optimizer]


def lr_find(module: pl.LightningModule, gpu_id: typing.Union[torch.device, int] = None,
            init_value: float = 1e-8, final_value: float = 10., beta: float = 0.98,
            max_steps: int = None) -> (typing.List[float], typing.List[float]):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = pathlib.Path(tmpdir) / 'model.pth'
        torch.save(module.state_dict(), save_path)
        train_dataloader = module.train_dataloader()

        if max_steps is None:
            num = len(train_dataloader) - 1
        else:
            num = min(len(train_dataloader) - 1, max_steps)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value

        avg_loss = 0.
        best_loss = 0.
        losses = []
        lrs = []

        optimizers = initialize_optimizers(module, lr)

        if gpu_id is not None:
            module = module.to(gpu_id)

        for batch_num, batch in enumerate(tqdm(train_dataloader, total=num), start=1):
            if gpu_id is not None:
                batch = transfer_batch_to_gpu(batch, gpu_id)
            loss = module.training_step(batch, batch_num)['loss']

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            if lr >= final_value:
                break

            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            losses.append(smoothed_loss)
            lrs.append(lr)

            loss.backward()
            optimizers = step_optimizers(optimizers)
            optimizers = zero_grad_optimizers(optimizers)

            # Update the lr for the next step
            lr *= mult
            optimizers = set_optimizer_lr(optimizers, lr)
        module.load_state_dict(torch.load(save_path))
    return lrs, losses


def plot_lr_curve(lrs: list, losses: list, truncate: bool = True, show=True) -> None:
    import matplotlib.pyplot as plt

    if truncate:
        _log_lrs = lrs[10:-5]
        _losses = losses[10:-5]
    else:
        _log_lrs = lrs
        _losses = losses
    plt.plot(_log_lrs, _losses)
    plt.xscale('log')
    if show:
        plt.show()


def initialize_optimizers(module: pl.LightningModule, lr: float) -> IterOptimizer:
    optimizers = get_optimizers_only(module)
    optimizers = set_optimizer_lr(optimizers, lr)
    optimizers = zero_grad_optimizers(optimizers)
    return optimizers


def set_optimizer_lr(optimizers: IterOptimizer, lr: float) -> IterOptimizer:
    for _optimizer in optimizers:
        for _param_group in _optimizer.param_groups:
            _param_group['lr'] = lr
    return optimizers


def zero_grad_optimizers(optimizers: IterOptimizer) -> IterOptimizer:
    for _optimizer in optimizers:
        _optimizer.zero_grad()
    return optimizers


def step_optimizers(optimizers):
    for _optimizer in optimizers:
        _optimizer.step()
    return optimizers


def get_optimizers_only(module: pl.LightningModule) -> IterOptimizer:
    optimizers = module.configure_optimizers()
    if isinstance(optimizers, Optimizer):
        return [optimizers]

    if len(optimizers) > 1:
        # either list of optimizers or list of optimizers + schedulers
        if isinstance(optimizers[1], Optimizer):
            return optimizers
        else:
            if isinstance(optimizers[0], Optimizer):
                return [optimizers[0]]
            else:
                return optimizers[0]
    else:
        # list with single optimizer
        assert isinstance(optimizers[0], Optimizer)
        return optimizers


def transfer_batch_to_gpu(batch: typing.Any, gpu_id: typing.Union[torch.device, int]) -> typing.Any:
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)):
        return batch.cuda(gpu_id)

    elif callable(getattr(batch, 'to', None)):
        return batch.to(torch.device('cuda', gpu_id))

    # when list
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = transfer_batch_to_gpu(x, gpu_id)
        return batch

    # when tuple
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = transfer_batch_to_gpu(x, gpu_id)
        return tuple(batch)

    # when dict
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = transfer_batch_to_gpu(v, gpu_id)

        return batch

    # nothing matches, return the value as is without transform
    return batch
