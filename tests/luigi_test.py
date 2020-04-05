import utils.luigi_wrapper as luigi
from utils.utils import *


#
class TestTask(luigi.Task):
    x = luigi.luigi.IntParameter()


if __name__ == '__main__':
    luigi.run_task(TestTask(), local_scheduler=get_from_config('luigi_local_scheduler'), delete_all=False)
