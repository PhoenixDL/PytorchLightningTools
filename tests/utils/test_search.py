import unittest
from pltools.utils.search import breadth_first_multiple_search


class TestSearch(unittest.TestCase):
    def setUp(self):
        self._example_dict = {
            "layer0_0": {"layer1_0": "l10", "layer1_1": "l11", "layer1_2": "l12"},
            "layer0_1": {"layer1_3": "l10", "layer4_1": "l11", "layer1_2": "l122"}, }

    def test_breadth_first_multiple_search(self):
        _key, _item = breadth_first_multiple_search(self._example_dict, "layer0_0")
        self.assertEqual(_item, [self._example_dict["layer0_0"]])
        self.assertEqual(_key, ["layer0_0"])

        _key, _item = breadth_first_multiple_search(self._example_dict, "layer1_1")
        self.assertEqual(_item, [self._example_dict["layer0_0"]["layer1_1"]])
        self.assertEqual(_key, ["layer0_0.layer1_1"])

        _key, _item = breadth_first_multiple_search(self._example_dict, "layer1_2")
        self.assertEqual(_item, [self._example_dict["layer0_0"]["layer1_2"],
                                 self._example_dict["layer0_1"]["layer1_2"]])
        self.assertEqual(_key, ["layer0_0.layer1_2", "layer0_1.layer1_2"])

        _key, _item = breadth_first_multiple_search(self._example_dict,
                                                    "layerX_X")
        self.assertFalse(_key)
        self.assertFalse(_item)


if __name__ == '__main__':
    unittest.main()