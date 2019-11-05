import unittest
import numpy as np

from pltools.data import LoadSample, LoadSampleLabel
from pltools.data.load_fn import norm_zero_mean_unit_std


class TestLoadsample(unittest.TestCase):
    @staticmethod
    def load_dummy_label(path):
        return {'label': 42}

    @staticmethod
    def load_dummy_data(path):
        return np.random.rand(1, 256, 256) * np.random.randint(2, 20) + \
            np.random.randint(20)

    def test_load_sample(self):
        # check loading of a single sample
        sample_fn = LoadSample({'data': ['data', 'data', 'data'],
                                'seg': ['data'],
                                'data2': ['data', 'data', 'data']},
                               self.load_dummy_data,
                               dtype={'seg': 'uint8'},
                               normalize=('data2',))
        sample = sample_fn('load')
        assert not np.isclose(np.mean(sample['data']), 0)
        assert not np.isclose(np.mean(sample['seg']), 0)
        assert sample['seg'].dtype == 'uint8'
        assert np.isclose(sample['data2'].max(), 1)
        assert np.isclose(sample['data2'].min(), -1)

    def test_load_sample_zero_mean_norm(self):
        # check different normalization function
        sample_fn = LoadSample({'data': ['data', 'data', 'data']},
                               self.load_dummy_data,
                               normalize=('data',),
                               norm_fn=norm_zero_mean_unit_std)
        sample = sample_fn('load')
        assert np.isclose(np.mean(sample['data']), 0)
        assert np.isclose(np.std(sample['data']), 1)

    def test_load_sample_label(self):
        # check label and loading of single sample
        sample_fn = LoadSampleLabel(
            {'data': ['data', 'data', 'data'], 'seg': ['data'],
             'data2': ['data', 'data', 'data']}, self.load_dummy_data,
            'label', self.load_dummy_label,
            dtype={'seg': 'uint8'}, normalize=('data2',))
        sample = sample_fn('load')
        assert not np.isclose(np.mean(sample['data']), 0)
        assert not np.isclose(np.mean(sample['seg']), 0)
        assert sample['seg'].dtype == 'uint8'
        assert np.isclose(sample['data2'].max(), 1)
        assert np.isclose(sample['data2'].min(), -1)
        assert sample['label'] == 42


if __name__ == '__main__':
    unittest.main()
