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

    def test_nested_get_key(self):
        val = self.cfg.nested_get_key(2)
        self.assertEqual(val, ['nest0.num'])

    def test_nested_get_key_fn(self):
        val = self.cfg.nested_get_key_fn(lambda x: isinstance(x, str))
        self.assertEqual(val, ["str"])

    def test_set_with_dot_str(self):
        self.cfg.set_with_dot_str('nest0.test_num', 123)
        self.assertEqual(self.cfg["nest0"]["test_num"], 123)

        self.cfg.set_with_dot_str('nest1.test_num', 234, create=True)
        self.assertEqual(self.cfg["nest1"]["test_num"], 234)

        with self.assertRaises(ValueError):
            self.cfg.set_with_dot_str('nest2.test_num', 234)

    def test_nested_get_first(self):
        val = self.cfg.nested_get_first("num")
        self.assertEqual(val, 2)

        val = self.cfg.nested_get_first("does_not_exist", default=False)
        self.assertFalse(val)

    def test_nested_contains(self):
        val = self.cfg.nested_contains("nested_num")
        self.assertTrue(val)

        val = self.cfg.nested_contains("nested_num2")
        self.assertFalse(val)

    def test_patch_dictconf(self):
        self.assertFalse(patch_dictconf())

    def test_get_with_dot_str(self):
        val = self.cfg.get_with_dot_str("nest0.num")
        self.assertEqual(val, 2)


if __name__ == '__main__':
    unittest.main()
