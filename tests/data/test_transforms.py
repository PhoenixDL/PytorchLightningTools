import unittest
import torch
import numpy as np
from pltools.data import ToTensor


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


if __name__ == '__main__':
    unittest.main()
