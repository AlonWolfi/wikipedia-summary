from unittest import TestCase
import pathlib

from utils.utils import *

class UtilsTest(TestCase):
    def test_task_done(self):
        test_name = 'test'
        task_done(test_name)
        path = pathlib.Path.cwd().parents[1] / get_config()['cache_dir'] / (test_name + '.done')
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.isfile(path))
        os.remove(path)
    def test_get_project_dir(self):
        self.assertEqual(get_project_dir(), pathlib.Path.cwd().parents[1])
