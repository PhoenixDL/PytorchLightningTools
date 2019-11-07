import unittest

from collections import OrderedDict
from omegaconf import DictConfig
from pltools.config import patch_dictconf


class TestOmegaConfDict(unittest.TestCase):
    def setUp(self) -> None:
        _d = OrderedDict()
        _d["str"] = "0"
        _d["nest0"] = {"num": 2, "nested_num": 0}
        _d["num"] = 1
        self.cfg = DictConfig(_d)

    def test_nested_get(self):
        val = self.cfg.nested_get("num")
        self.assertEqual(val, [2, 1])

    def test_nested_get_first(self):
        val = self.cfg.nested_get_first("num")
        self.assertEqual(val, 2)

    def test_nested_contains(self):
        val = self.cfg.nested_contains("nested_num")
        self.assertTrue(val)

        val = self.cfg.nested_contains("nested_num2")
        self.assertFalse(val)

    def test_patch_dictconf(self):
        self.assertFalse(patch_dictconf())


if __name__ == '__main__':
    unittest.main()
