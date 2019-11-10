import unittest
import os
import numpy as np
from pltools.config import HydraDecoder, Config
from functools import partial


class TestHydraDecoder(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self._original_dict = Config.create_from_file(
            os.path.join(current_dir, '_src', 'config.yml'))
        self._decoded_dict = HydraDecoder()(self._original_dict)

    def test_function(self):
        self.assertTrue(isinstance(self._decoded_dict["exp.im_an_function"],
                                   partial))
        self.assertDictEqual(self._decoded_dict["exp.im_an_function"].keywords,
                             {"axis": 0})

    def test_class(self):
        self.assertTrue(isinstance(self._decoded_dict["exp.im_an_object"],
                                   HydraDecoder))

    def test_functionref(self):
        self.assertEqual(self._decoded_dict["exp.im_a_functionref"], np.min)

    def test_classref(self):
        self.assertEqual(self._decoded_dict["exp.im_a_classref"], HydraDecoder)


if __name__ == '__main__':
    unittest.main()
