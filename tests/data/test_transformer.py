import unittest
import torch
import math
import numpy as np

from torch.utils.data import DataLoader

from batchgenerators.transforms import ZeroMeanUnitVarianceTransform

from tests import DummyDataset
from pltools.data import Transformer, ToTensor


class TestToTensor(unittest.TestCase):
    def setUp(self):
        self.data = {'data': np.random.rand(1, 3, 128, 128).astype(np.int32),
                     'label': np.random.rand(1, 1, 128, 128).astype(np.int32)}

    def test_to_tensor(self):
        trafo = ToTensor()
        self._check_trafo(trafo,
                          keys=('data', 'label'),
                          dtypes=(('data', torch.int32),
                                  ('label', torch.int32)))

    def test_to_tensor_keys(self):
        trafo = ToTensor(keys='data')
        self._check_trafo(trafo,
                          keys=('data',),
                          dtypes=(('data', torch.int32),))

    def test_to_tensor_dtypes(self):
        trafo = ToTensor(keys=('data',),
                         dtypes=(('data', torch.float32),))
        self._check_trafo(trafo,
                          keys=('data',),
                          dtypes=(('data', torch.float32),))

    def _check_trafo(self, trafo, keys, dtypes):
        torch_data = trafo(**self.data)

        for _key, _item in torch_data.items():
            if _key in keys:
                self.assertTrue(isinstance(_item, torch.Tensor))
            else:
                self.assertTrue(isinstance(_item, np.ndarray))

        for _dtype in dtypes:
            self.assertTrue(torch_data[_dtype[0]].dtype == _dtype[1])


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.dset = DummyDataset()

    def test_transformer_getitem(self):
        for sample in self.dset:
            self.assertFalse(np.isclose(sample['data'].mean(), 0))

        transformer = Transformer(self.dset, [ZeroMeanUnitVarianceTransform()])

        for sample in transformer:
            self.assertTrue(np.isclose(sample['data'].mean(), 0))

    def test_transformer_transforms(self):
        transformer = Transformer(self.dset, [ZeroMeanUnitVarianceTransform()])
        transformer.transforms = []

        for sample in transformer:
            self.assertFalse(np.isclose(sample['data'].mean(), 0))

    def test_integration_transformer_dataloading(self):
        batch_size = 4
        last_batch_size = len(self.dset) % batch_size
        transformer = Transformer(self.dset, [ZeroMeanUnitVarianceTransform()])
        dataloader = DataLoader(
            transformer, batch_size=batch_size, num_workers=1)

        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch != math.floor(len(transformer) / batch_size):
                _size = batch_size
            else:
                _size = last_batch_size

            self.assertTupleEqual(
                sample_batched['data'].size(), (_size, 3, 128, 128))
            self.assertTupleEqual(
                sample_batched['label'].size(), (_size, 3))
            self.assertEqual(len(sample_batched['id']), _size)


if __name__ == '__main__':
    unittest.main()
