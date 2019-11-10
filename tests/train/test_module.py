import unittest
from torch.utils.data import DataLoader

from pltools.data import Transformer
from pltools.train import PLTModule
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


class TestPLTModule(unittest.TestCase):
    def setUp(self):
        config = Config({"dataloader": {"batch_size": 1, "num_workers": 1}})
        self.transformer = Transformer(DummyDataset(), [])
        self.module = PLTModule(config, DummyModel())

    def test_get_dataloading_kwargs_dataloader(self):
        module = PLTModule(Config({"dataloader": {"batch_size": 8, "num_workers": 4}}),
                           DummyModel())
        kwargs = module.get_dataloading_kwargs('train_dataloader')
        self.assertDictEqual(kwargs, {"batch_size": 8, "num_workers": 4})

    def test_get_dataloading_kwargs_train_dataloader(self):
        module = PLTModule(
            Config({"dataloader": {"batch_size": 4, "num_workers": 2},
                    "train_dataloader": {"batch_size": 8, "num_workers": 4}}),
            DummyModel())

        train_kwargs = module.get_dataloading_kwargs('train_dataloader')
        self.assertDictEqual(train_kwargs, {"batch_size": 8, "num_workers": 4})

        val_kwargs = module.get_dataloading_kwargs('val_dataloader')
        self.assertDictEqual(val_kwargs, {"batch_size": 4, "num_workers": 2})

    def test_get_dataloading_kwargs_empty(self):
        module = PLTModule(Config(), DummyModel())
        kwargs = module.get_dataloading_kwargs('train_dataloader')
        self.assertFalse(kwargs)

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
        transform_name = f"{name}_transformer"

        setattr(self.module, transform_name, None)
        no_transforms_dataloader = getattr(self.module, loader_name)()

        self.assertIsNone(no_transforms_dataloader)

    def check_dataloader(self, name: str):
        loader_name = f"{name}_dataloader"
        transform_name = f"{name}_transformer"

        setattr(self.module, transform_name, self.transformer)
        loader_fn = getattr(self.module, loader_name)
        transforms_dataloader = loader_fn()

        if name in ["val", "test"]:
            transforms_dataloader = transforms_dataloader[0]

        self.assertIsInstance(transforms_dataloader, DataLoader)
        self.assertEqual(transforms_dataloader.batch_size, 1)
        self.assertEqual(transforms_dataloader.num_workers, 1)

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
