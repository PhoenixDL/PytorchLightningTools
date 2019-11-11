import unittest
from torch.utils.data import DataLoader

from pltools.train import Module
from pltools.config import Config

from tests import DummyDataset, DummyModel


class DummyTransform:
    idx = 0

    def __init__(self):
        super().__init__()
        self.called = False
        self.idx = DummyTransform.idx
        DummyTransform.idx += 1

    def __call__(self, data):
        self.called = True
        return data


class TestModule(unittest.TestCase):
    def setUp(self):
        config = Config({"dataloader": {"batch_size": 1, "num_workers": 1}})
        self.dataloader = DataLoader(DummyDataset())
        self.module = Module(config, DummyModel())

    def test_train_dataloader(self):
        with self.assertRaises(NotImplementedError):
            self.check_dataloader_super("train")
        self.check_dataloader("train")

    def test_val_dataloader(self):
        self.check_dataloader_super("val")
        self.check_dataloader("val")

    def test_test_dataloader(self):
        self.check_dataloader_super("test")
        self.check_dataloader("test")

    def check_dataloader_super(self, name: str):
        loader_name = f"{name}_dataloader"
        data_name = f"{name}_data"

        setattr(self.module, data_name, None)
        no_transforms_dataloader = getattr(self.module, loader_name)()

        self.assertIsNone(no_transforms_dataloader)

    def check_dataloader(self, name: str):
        loader_name = f"{name}_dataloader"
        data_name = f"{name}_data"

        setattr(self.module, data_name, self.dataloader)
        loader_fn = getattr(self.module, loader_name)
        transforms_dataloader = loader_fn()

        if name in ["val", "test"]:
            transforms_dataloader = transforms_dataloader[0]

        self.assertIsInstance(transforms_dataloader, DataLoader)
        self.assertEqual(transforms_dataloader.batch_size, 1)
        self.assertEqual(transforms_dataloader.num_workers, 0)

    def check_enable_tta(self):
        trafos = [DummyTransform(), DummyTransform(), DummyTransform()]
        inverse_trafos = [DummyTransform(), DummyTransform(), DummyTransform()]

        self.module.enable_tta(trafos=trafos,
                               inverse_trafos=inverse_trafos,
                               tta_reduce=lambda x: x)
        output = self.module.forward(0)
        for _trafo in [*trafos, *inverse_trafos]:
            self.assertTrue(_trafo.called)
        self.assertEqual(output, [0, 1, 2, 3])

    def test_tta(self):
        self.assertFalse(self.module.disable_tta())
        self.check_enable_tta()
        self.assertTrue(self.module.disable_tta())

        output = self.module.forward(0)
        self.assertEqual(output, self.module.model.call_count)


if __name__ == '__main__':
    unittest.main()
