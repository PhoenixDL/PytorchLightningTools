import unittest

import numpy as np

from pltools.data import (
    CacheDataset, LazyDataset, LoadSample, LoadSampleLabel,
    CacheDatasetID, LazyDatasetID)
from pltools.data.load_fn import norm_zero_mean_unit_std

# TODO: additional testing of ConcatDataset, BlankDataset


class LoadDummySample:
    def __call__(self, path, *args, **kwargs):
        data = {'data': np.random.rand(1, 256, 256),
                'label': np.random.randint(2),
                'id': f"sample{path}"}
        return data


class TestBaseDataset(unittest.TestCase):
    def setUp(self):
        self.paths = list(range(10))

    def test_cache_dataset(self):
        dataset = CacheDataset(self.paths, LoadDummySample(),
                               label_load_fct=None)
        self.assertEqual(len(dataset), 10)
        self.check_dataset_access(dataset, [0, 5, 9])
        self.check_dataset_outside_access(dataset, [10, 20])
        self.check_dataset_iter(dataset)

    def test_cache_dataset_extend(self):
        def load_mul_sample(path) -> list:
            return [LoadDummySample()(path, None)] * 4

        dataset = CacheDataset(self.paths, load_mul_sample,
                               mode='extend')
        self.assertEqual(len(dataset), 40)
        self.check_dataset_access(dataset, [0, 20, 39])
        self.check_dataset_outside_access(dataset, [40, 45])
        self.check_dataset_iter(dataset)

    def test_lazy_dataset(self):
        dataset = LazyDataset(self.paths, LoadDummySample(),
                              label_load_fct=None)
        self.assertEqual(len(dataset), 10)
        self.check_dataset_access(dataset, [0, 5, 9])
        self.check_dataset_outside_access(dataset, [10, 20])
        self.check_dataset_iter(dataset)

    def check_dataset_access(self, dataset, inside_idx):
        try:
            for _i in inside_idx:
                a = dataset[_i]
        except BaseException:
            self.assertTrue(False)

    def check_dataset_outside_access(self, dataset, outside_idx):
        for _i in outside_idx:
            with self.assertRaises(IndexError):
                a = dataset[_i]

    def check_dataset_iter(self, dataset):
        try:
            j = 0
            for i in dataset:
                self.assertIn('data', i)
                self.assertIn('label', i)
                j += 1
            assert j == len(dataset)
        except BaseException:
            raise AssertionError('Dataset iteration failed.')


class TestDatasetID(unittest.TestCase):
    def test_load_dummy_sample(self):
        load_fn = LoadDummySample()
        sample0 = load_fn(None, None)
        self.assertIn("data", sample0)
        self.assertIn("label", sample0)
        self.assertTrue("id", "sample0")

        sample1 = load_fn(None, None)
        self.assertIn("data", sample1)
        self.assertIn("label", sample1)
        self.assertTrue("id", "sample1")

    def check_dataset(
            self,
            dset_cls,
            num_samples,
            expected_len,
            debug_num,
            **kwargs):
        load_fn = LoadDummySample()
        dset = dset_cls(list(range(num_samples)), load_fn,
                        debug_num=debug_num, **kwargs)
        self.assertEqual(len(dset), expected_len)

    def test_base_cache_dataset(self):
        self.check_dataset(CacheDataset, num_samples=20,
                           expected_len=20, debug_num=10)

    def test_base_lazy_dataset_debug_off(self):
        self.check_dataset(LazyDataset, num_samples=20,
                           expected_len=20, debug_num=10)

    def test_cachedataset_id(self):
        load_fn = LoadDummySample()
        dset = CacheDatasetID(list(range(10)), load_fn,
                              id_key="id", cache_ids=False)
        self.check_dset_id(dset)

    def test_lazydataset_id(self):
        load_fn = LoadDummySample()
        dset = LazyDatasetID(list(range(10)), load_fn,
                             id_key="id", cache_ids=False)
        self.check_dset_id(dset)

    def check_dset_id(self, dset):
        idx = dset.get_index_by_id("sample1")
        self.assertTrue(idx, 1)

        with self.assertRaises(KeyError):
            idx = dset.get_index_by_id("sample10")

        sample5 = dset.get_sample_by_id("sample5")
        self.assertTrue(sample5["id"], 5)

        dset.cache_ids()
        sample6 = dset.get_sample_by_id("sample6")
        self.assertTrue(sample6["id"], 6)


if __name__ == "__main__":
    unittest.main()
