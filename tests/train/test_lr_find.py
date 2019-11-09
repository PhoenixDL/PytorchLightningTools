import unittest
from unittest.mock import Mock, patch
import pytorch_lightning as pl
import torch
import logging
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from tests._data_utils import DummyModel, DummyModule
from pltools.train.lr_find import (
    transfer_batch_to_gpu, get_optimizers_only, step_optimizers, zero_grad_optimizers,
    set_optimizer_lr, initialize_optimizers, lr_find, plot_lr_curve)


class TestLRFind(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizers = [SGD(DummyModel().parameters(), lr=0.1, momentum=0.9),
                           SGD(DummyModel().parameters(), lr=0.1, momentum=0.9),
                           ]

    def test_lr_find(self):
        length = 1000
        log_lrs, losses = lr_find(DummyModule(length=length))
        self.assertEqual(len(log_lrs), length-1)
        self.assertEqual(len(losses), length-1)

    def test_lr_find_max_step(self):
        length = 1000
        max_steps = 100
        log_lrs, losses = lr_find(DummyModule(length=length), max_steps=max_steps)
        self.assertEqual(len(log_lrs), max_steps)
        self.assertEqual(len(losses), max_steps)

    def test_lr_find_gpu(self):
        if torch.cuda.is_available():
            length = 1000
            log_lrs, losses = lr_find(DummyModule(length=length), gpu_id=0)
            self.assertEqual(len(log_lrs), length-1)
            self.assertEqual(len(losses), length-1)
        else:
            logging.warning("test_lr_find_gpu was skipped because of missing GPU")

    def test_transfer_batch_to_gpu(self):
        def check_float_cuda():
            t = torch.Tensor([0])
            t = transfer_batch_to_gpu(t, 0)
            self.assertEqual(t.device, torch.device(type='cuda', index=0))

        def check_float_to():
            t = torch.Tensor([0])
            t = transfer_batch_to_gpu(t, 0)
            self.assertEqual(t.device, torch.device(type='cuda', index=0))

        def check_list():
            t = [torch.Tensor([0]), torch.Tensor([0])]
            t = transfer_batch_to_gpu(t, 0)
            for _t in t:
                self.assertEqual(_t.device, torch.device(type='cuda', index=0))

        def check_tuple():
            t = (torch.Tensor([0]), torch.Tensor([0]))
            t = transfer_batch_to_gpu(t, 0)
            for _t in t:
                self.assertEqual(_t.device, torch.device(type='cuda', index=0))

        def check_dict():
            t = {"a": torch.Tensor([0]), "b": torch.Tensor([0])}
            t = transfer_batch_to_gpu(t, 0)
            for k, _t in t.items():
                self.assertEqual(_t.device, torch.device(type='cuda', index=0))

        def check_string():
            t = transfer_batch_to_gpu("hi", 0)
            self.assertEqual(t, "hi")

        if torch.cuda.is_available():
            check_float_cuda()
            check_float_to()
            check_list()
            check_tuple()
            check_dict()
            check_string()
        else:
            logging.warning("test_transfer_batch_to_gpu was skipped because of missing GPU")

    def test_get_optimizers_only(self):
        module = pl.LightningModule()

        module.configure_optimizers = Mock(return_value=self.optimizers)
        optim = get_optimizers_only(module)
        self.check_optimizer_iter(optim, 2)

        module.configure_optimizers = Mock(return_value=self.optimizers[0])
        optim = get_optimizers_only(module)
        self.check_optimizer_iter(optim, 1)

        module.configure_optimizers = Mock(return_value=(self.optimizers, []))
        optim = get_optimizers_only(module)
        self.check_optimizer_iter(optim, 2)

        module.configure_optimizers = Mock(return_value=(self.optimizers[0], []))
        optim = get_optimizers_only(module)
        self.check_optimizer_iter(optim, 1)

    def check_optimizer_iter(self, optimizers, length):
        for _o in optimizers:
            self.assertIsInstance(_o, Optimizer)

        self.assertEqual(len(optimizers), length)

    def test_step_optimizers(self):
        for _optimizer in self.optimizers:
            _optimizer.step = Mock()

        self.optimizers = step_optimizers(self.optimizers)

        for _optimizer in self.optimizers:
            _optimizer.step.assert_called_once()

    def test_zero_grad_optimizers(self):
        for _optimizer in self.optimizers:
            _optimizer.zero_grad = Mock()

        self.optimizers = zero_grad_optimizers(self.optimizers)

        for _optimizer in self.optimizers:
            _optimizer.zero_grad.assert_called_once()

    def test_set_optimizer_lr(self):
        optimizers = set_optimizer_lr(self.optimizers, 100)
        for _optimizer in optimizers:
            for _param_group in _optimizer.param_groups:
                self.assertEqual(_param_group['lr'], 100)

    def test_initialize_optimizers(self):
        module = pl.LightningModule()
        module.configure_optimizers = Mock(return_value=[self.optimizers, []])

        for _optimizer in self.optimizers:
            _optimizer.zero_grad = Mock()

        optimizers = initialize_optimizers(module, 100)

        for _optimizer in optimizers:
            _optimizer.zero_grad.assert_called_once()

        for _optimizer in optimizers:
            for _param_group in _optimizer.param_groups:
                self.assertEqual(_param_group['lr'], 100)

    def test_plot_lr_curve(self):
        plot_lr_curve([0.001, 0.01, 0.1, 1], [100, 10, 1, 10], truncate=False, show=False)
        plot_lr_curve([1] * 20, [1] * 20, show=False)


if __name__ == '__main__':
    unittest.main()
