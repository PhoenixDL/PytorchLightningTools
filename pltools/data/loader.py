from typing import Callable, Mapping, Iterable, Sequence
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataloader import \
    _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter as \
    __MultiProcessingDataLoaderIter
from pltools import get_current_debug_mode
from functools import partial


class DataLoader(_DataLoader):
    """
    A Dataloader introducing batch-transforms, numpy seeds for worker processes
    and compatibility to the debug mode

    Note
    ----
    For Reproducibility numpy and pytorch must be seeded in the main process,
    as these frameworks will be used to generate their own seeds for each
    worker.

    Note
    ----
    ``len(dataloader)`` heuristic is based on the length of the sampler used.
    When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
    an infinite sampler is used, whose :meth:`__len__` is not
    implemented, because the actual length depends on both the
    iterable as well as multi-process loading configurations. So one
    should not query this method unless they work with a map-style
    dataset. See `Dataset Types`_ for more details on these two types
    of datasets.

    Warning
    -------
    If the ``spawn`` start method is used, :attr:`worker_init_fn`
    cannot be an unpicklable object, e.g., a lambda function. See
    :ref:`multiprocessing-best-practices` on more details related
    to multiprocessing in PyTorch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 batch_transforms=None, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 auto_convert=True):
        """
        Parameters
        ----------
        dataset : Dataset
            dataset from which to load the data
        batch_size : int, optional
            how many samples per batch to load (default: ``1``).
        shuffle : bool, optional
            set to ``True`` to have the data reshuffled at every epoch
            (default: ``False``)
        batch_transforms : callable, optional
            transforms which can be applied to a whole batch.
            Usually this accepts either mappings or sequences and returns the
            same type containing transformed elements
        sampler : torch.utils.data.Sampler, optional
            defines the strategy to draw samples from
            the dataset. If specified, :attr:`shuffle` must be ``False``.
        batch_sampler : torch.utils.data.Sampler, optional
            like :attr:`sampler`, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers : int, optional
            how many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn : callable, optional
            merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory : bool, optional
            If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data
            elements are a custom type, or your :attr:`collate_fn` returns a
            batch that is a custom type, see the example below.
        drop_last : bool, optional
            set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If ``False`` and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller. (default: ``False``)
        timeout : numeric, optional
            if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn : callable, optional
            If not ``None``, this will be called on each
            worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and
            before data loading. (default: ``None``)
        auto_convert : bool, optional
            if set to ``True``, the batches will always be transformed to
            torch.Tensors, if possible. (default: ``True``)
        """

        super().__init__(dataset=dataset, batch_size=batch_size,
                         shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers,
                         collate_fn=collate_fn, pin_memory=pin_memory,
                         drop_last=drop_last, timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context)

        self.collate_fn = BatchTransformer(self.collate_fn, batch_transforms,
                                           auto_convert)

    def __iter__(self):
        if self.num_workers == 0 or get_current_debug_mode():
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)


class BatchTransformer(object):
    """
    A callable wrapping the collate_fn to enable transformations on a
    batch-basis.
    """

    def __init__(self, collate_fn: Callable, transforms: Callable = None,
                 auto_convert=True):
        """
        Parameters
        ----------
        collate_fn : callable, optional
            merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        transforms : callable, optional
            transforms which can be applied to a whole batch.
            Usually this accepts either mappings or sequences and returns the
            same type containing transformed elements
        auto_convert : bool, optional
            if set to ``True``, the batches will always be transformed to
            torch.Tensors, if possible. (default: ``True``)
        """

        self._collate_fn = collate_fn
        self._transforms = transforms
        self._auto_convert = auto_convert

    def __call__(self, *args, **kwargs):
        batch = self._collate_fn(*args, **kwargs)

        if self._transforms is not None:

            if isinstance(batch, Mapping):
                batch = self._transforms(**batch)
            elif isinstance(batch, Sequence):
                batch = self._transforms(*batch)
            else:
                batch = self._transforms(batch)

        if self._auto_convert:
            batch = default_convert(batch)

        return batch


class _MultiProcessingDataLoaderIter(__MultiProcessingDataLoaderIter):
    # NOTE [ Numpy Seeds ]
    # This class is a subclass of
    # ``torch.utils.data.dataloader._MultiProcessingDataLoaderIter``` and only
    # adds some additional logic to provide differnt seeds for numpy in
    # each worker. These seeds are based on a base seed, which itself get's
    # generated by numpy. So to ensure reproducibility, numpy must be seeded
    # in the main process.
    def __init__(self, loader):

        try:
            import numpy as np
            npy_seed = np.random.randint(0, (2 ** 32) - (1 + loader.num_workers))
        except ImportError:
            # we don't generate a numpy seed here with torch, since we don't
            # need one; if the import fails in the main process it should
            # also fail in child processes
            npy_seed = None

        old_worker_init = loader.worker_init_fn

        new_worker_init_fn = partial(_seed_npy_before_worker_init,
                                     seed=npy_seed,
                                     worker_init_fn=old_worker_init)
        loader.worker_init_fn = new_worker_init_fn

        super().__init__(loader)

        # reset worker_init_fn once the workers have been startet to reset
        # to original state for next epoch
        loader.worker_init_fn = old_worker_init


def _seed_npy_before_worker_init(worker_id, seed, worker_init_fn=None):
    """
    Wrapper Function to wrap the existing worker_init_fn and seed numpy before
    calling the actual ``worker_init_fn``

    Parameters
    ----------
    worker_id : int
        the number of the worker
    seed : int32
        the base seed in a range of [0, 2**32 - (1 + ``num_workers``)].
        The range ensures, that the whole seed, which consists of the base
        seed and the ``worker_id``, can still be represented as a unit32,
        as it needs to be for numpy seeding
    worker_init_fn : callable, optional
        will be called with the ``worker_id`` after seeding numpy if it is not
        ``None``
    """
    try:
        import numpy as np
        np.random.seed(seed + worker_id)
    except ImportError:
        pass

    if worker_init_fn is not None:
        return worker_init_fn(worker_id)
