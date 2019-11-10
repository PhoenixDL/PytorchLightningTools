import unittest
import os
import copy
import argparse

from pltools.config import Config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_cls = Config
        self.example_dict = {
            "shallowStr": "a",
            "shallowNum": 1,
            "deep": {"deepStr": "b", "deepNum": 2},
            "nestedListOrig": [{"dictList": [1, 2, 3]}],
        }
        self.update_dict = {
            "deep": {"deepStr": "c"},
            "shallowNew": 3,
            "deepNew": {"newNum": 4},
            "nestedList": [{"dictList": [1, 2, 3]}],
            "nestedList2": [{"dictList": [1, 2, 3]}],
        }

    def test_config_access(self):
        # initialization from dict
        cf = self.config_cls(self.example_dict)
        self.assertEqual(cf["shallowStr"], self.example_dict["shallowStr"])
        self.assertEqual(cf["shallowNum"], self.example_dict["shallowNum"])

        # check if parameters were written correctly
        self.assertEqual(cf["deep"]["deepStr"],
                         self.example_dict["deep"]["deepStr"])
        self.assertEqual(cf["deep"]["deepNum"],
                         self.example_dict["deep"]["deepNum"])

        # check deep acces with operators
        self.assertEqual(cf["deep.deepStr"],
                         self.example_dict["deep"]["deepStr"])
        self.assertEqual(cf.deep.deepNum,
                         self.example_dict["deep"]["deepNum"])

        # empty initialization
        cf = self.config_cls()

        # set shallow attributes
        cf.shallowString = "string"
        cf.shallowNum = 1
        cf.deep = {}
        cf.deep.string = "deepString"
        cf.deep.num = 2

        cf["shallowString2"] = "string2"
        cf["shallowNum2"] = 1
        cf["deep.string2"] = "deepString2"
        cf["deep.num2"] = 2

        # check if parameters were written correctly
        self.assertEqual(cf["shallowString"], "string")
        self.assertEqual(cf["shallowNum"], 1)
        self.assertEqual(cf["deep.string"], "deepString")
        self.assertEqual(cf["deep.num"], 2)

        self.assertEqual(cf["shallowString2"], "string2")
        self.assertEqual(cf["shallowNum2"], 1)
        self.assertEqual(cf["deep.string2"], "deepString2")
        self.assertEqual(cf["deep.num2"], 2)

        # check contains operator
        self.assertTrue("shallowString" in cf)
        self.assertTrue("shallowString2" in cf)
        self.assertTrue("deep.string" in cf)
        self.assertTrue("deep.string2" in cf)

        warning_msg = ("The key 5 is not a string, but a <class 'int'>. "
                       "This may lead to unwanted behavior!")
        with self.assertWarns(RuntimeWarning, msg=warning_msg):
            cf[5] = 10

    def test_config_access_with_non_existing_keys(self):
        cf = self.config_cls(self.example_dict)

        with self.assertRaises(KeyError):
            cf["unknown_key"]

        with self.assertRaises(KeyError):
            cf["shallowStr.unknown_key"]

    def test_update(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        with self.assertRaises(ValueError):
            cf.update(self.update_dict)

        # update with overwrite
        cf.update(self.update_dict, overwrite=True)
        self.assertEqual(cf["deep.deepStr"],
                         self.update_dict["deep"]["deepStr"])

        # add new values
        self.assertEqual(cf["shallowNew"],
                         self.update_dict["shallowNew"])
        self.assertEqual(cf["deepNew.newNum"],
                         self.update_dict["deepNew"]["newNum"])

        # check for shallow copy
        cf["nestedList"][0]["dictList"][0] = 10
        self.assertEqual(self.update_dict["nestedList"][0]["dictList"][0],
                         cf["nestedList"][0]["dictList"][0])

        # check for deepcopy
        cf.update(self.update_dict, overwrite=True, deepcopy=True)
        cf["nestedList2"][0]["dictList"][0] = 10
        self.assertNotEqual(self.update_dict["nestedList2"][0]["dictList"][0],
                            cf["nestedList2"][0]["dictList"][0])

        # check for no error when only updating nested keys
        cf = self.config_cls.create_from_dict(self.example_dict)
        update_dict = copy.deepcopy(self.update_dict)
        update_dict["deep"].pop("deepStr")
        update_dict["deep"]["deepStr2"] = "deepStr2"
        cf.update(update_dict)
        self.assertEqual(cf["deep.deepStr2"],
                         update_dict["deep"]["deepStr2"])

    def test_dump_and_load(self):
        pass
        # cf = self.config_cls.create_from_dict(self.example_dict)
        # path = os.path.join(".", "test_config.yaml")
        # # # check dump
        # # cf.dump(path)
        #
        # # check load
        # cf_loaded = self.config_cls()
        # cf_loaded.load(path)
        # self.assertDictEqual(cf, cf_loaded)
        #
        # cf_loaded_file = self.config_cls.create_from_file(path)
        # self.assertDictEqual(cf, cf_loaded_file)

        # # check dump
        # cf_string = cf.dumps()

        # # check load
        # cf_loaded = self.config_cls()
        # cf_loaded.loads(cf_string)
        # self.assertDictEqual(cf, cf_loaded)
        #
        # cf_loaded_str = self.config_cls.create_from_str(cf_string)
        # self.assertDictEqual(cf, cf_loaded_str)

    def test_copy(self):
        cf = self.config_cls.create_from_dict(self.example_dict)

        # check for shallow copy
        cf_shallow = copy.copy(cf)
        cf_shallow["nestedListOrig"][0]["dictList"][0] = 10
        self.assertEqual(cf["nestedListOrig"][0]["dictList"][0],
                         cf_shallow["nestedListOrig"][0]["dictList"][0])

        # check for deepcopy
        cf_deep = copy.deepcopy(cf)
        cf_deep["nestedListOrig"][0]["dictList"][0] = 20
        self.assertNotEqual(cf["nestedListOrig"][0]["dictList"][0],
                            cf_deep["nestedListOrig"][0]["dictList"][0])

    def test_create_from_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p1')
        parser.add_argument('--param2')
        cf1 = self.config_cls.create_from_argparse(
            parser, args=['-p1', 'parameter1', '--param2', 'parameter2'])
        self.assertEqual(cf1['p1'], 'parameter1')
        self.assertEqual(cf1['param2'], 'parameter2')

        args = parser.parse_args(
            ['-p1', 'parameter1', '--param2', 'parameter2'])
        self.assertEqual(cf1['p1'], 'parameter1')
        self.assertEqual(cf1['param2'], 'parameter2')

    def test_internal_type(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        self.assertTrue(isinstance(cf["deep"], self.config_cls))


if __name__ == '__main__':
    unittest.main()
