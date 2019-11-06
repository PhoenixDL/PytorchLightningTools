import os
import unittest
import tempfile
from unittest.mock import Mock, call

from pltools.data import DataContainer
from pltools.scheduler import kfold_scheduler
from tests import DummyDataset


class TestKFoldScheduler(unittest.TestCase):
    def test_kfold_scheduler(self):
        def single_run(dsets, mock):
            self.assertIn('train', dsets)
            self.assertIn('val', dsets)
            current_fold = int(os.path.basename(os.getcwd())[-1])
            mock(current_fold)

        container = DataContainer(DummyDataset())
        folds = [{"train": [0], "val": [1]}] * 5

        mock = Mock()
        initial_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as td:
            os.chdir(td)
            kfold_scheduler(single_run, container.kfold_by_index(folds), mock)
        os.chdir(initial_cwd)

        calls = [call(0), call(1), call(2), call(3), call(4)]
        mock.assert_has_calls(calls)


if __name__ == '__main__':
    unittest.main()
