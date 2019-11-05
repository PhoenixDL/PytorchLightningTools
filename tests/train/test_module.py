import unittest
from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from pltools.data import Transformer
from pltools.train import PLTModule

from tests import DummyDataset, Config


class TestPLTModule(unittest.TestCase):
    def setUp(self):
        self.transformer = Transformer(DummyDataset(), [])
        self.module = PLTModule(
            Config(dataloader={"batch_size": 1, "num_workers": 1}),
            MagicMock())

    def test_get_dataloading_kwargs_dataloader(self):
        module = PLTModule(
            Config(dataloader={"batch_size": 8, "num_workers": 4}),
            MagicMock())
        kwargs = module.get_dataloading_kwargs('train_dataloader')
        self.assertDictEqual(kwargs, {"batch_size": 8, "num_workers": 4})
    
    def test_get_dataloading_kwargs_train_dataloader(self):
        module = PLTModule(
            Config(train_dataloader={"batch_size": 8, "num_workers": 4},
                   dataloader={"batch_size": 4, "num_workers": 2}),
                   MagicMock())
        
        train_kwargs = module.get_dataloading_kwargs('train_dataloader')
        self.assertDictEqual(train_kwargs, {"batch_size": 8, "num_workers": 4})

        val_kwargs = module.get_dataloading_kwargs('val_dataloader')
        self.assertDictEqual(val_kwargs, {"batch_size": 4, "num_workers": 2})
    
    def test_get_dataloading_kwargs_empty(self):
        module = PLTModule(Config(), MagicMock())
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
        loaderName = f"{name}_dataloader"
        transformName = f"{name}_transformer"

        setattr(self.module, transformName, None)
        no_tranforms_dataloader = getattr(self.module, loaderName)()

        self.assertIsNone(no_tranforms_dataloader)

    def check_dataloader(self, name: str):
        loaderName = f"{name}_dataloader"
        transformName = f"{name}_transformer"

        setattr(self.module, transformName, self.transformer)
        transforms_dataloader = getattr(self.module, loaderName)()

        self.assertIsInstance(transforms_dataloader, DataLoader)
        self.assertEqual(transforms_dataloader.batch_size, 1)
        self.assertEqual(transforms_dataloader.num_workers, 1)


if __name__ == '__main__':
    unittest.main()
