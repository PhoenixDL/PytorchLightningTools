import unittest
from unittest.mock import Mock
import nevergrad as ng
from pltools.config import Config

from pltools.scheduler.hpsearch import get_instrumentation_from_config, update_config_with_dot_str
from pltools.scheduler.hpsearch import hyperparameter_search


class TestHPSearch(unittest.TestCase):
    def setUp(self):
        self.cfg = Config({"variable": {
            "arg1": ng.var.OrderedDiscrete(["a", "b"]),
            "arg2": ng.var.OrderedDiscrete(["a", "b"])
        },
            "fixed": {"epochs": 100, "num_workers": 4}})

    def test_get_instrumentation_from_config(self):
        instr = get_instrumentation_from_config(self.cfg)
        self.assertEqual(len(instr.variables), 2)

    def test_update_config_with_dot_str(self):
        params = {"variable.arg1": "a", "variable.arg2": "b"}
        update_config_with_dot_str(self.cfg, params)
        self.assertEqual(self.cfg.variable.arg1, "a")
        self.assertEqual(self.cfg.variable.arg2, "b")

    def test_hyperparameter_search(self):
        singel_run = Mock(return_value=1)
        recommendation = hyperparameter_search(self.cfg, singel_run,
                                               ng.optimizers.RandomSearch, budget=5)
        singel_run.assert_called()


if __name__ == '__main__':
    unittest.main()
