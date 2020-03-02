import luigi
from luigi import LocalTarget
from luigi import Parameter, BoolParameter
from luigi import build
import shutil

from utils.utils import *


class Task(luigi.Task):
    DEBUG = luigi.BoolParameter(significant=True, default=get_from_config('DEBUG'))
    DATAFRAME = luigi.BoolParameter(significant=True, default=get_from_config('DATAFRAME'))

    @classmethod
    def __get_task_done_path(cls):
        return get_project_dir() / 'cache' / (cls.__name__ + '.done')

    def output(self):
        return LocalTarget(self.__get_task_done_path())

    @classmethod
    def get_output(cls):
        ins = cls()
        output_path = ins.output().path
        print(f"'{cls.__name__}' output's path is {output_path}")
        return read_data(output_path)

    @classmethod
    def task_done(cls):
        file_path = cls.__get_task_done_path()
        save_data('', file_path)

    def run(self):
        self.task_done()


def run_task(task: Task, local_scheduler: bool = False, delete_all: bool = False):
    if get_from_config('DELETE_CACHE'):
        if delete_all:
            shutil.rmtree(get_project_dir() / 'cache')
        else:
            output_path = task.output(task).path
            validate_path(output_path)
            if os.path.exists(output_path):
                os.remove(output_path)
    luigi.build(
        [
            task()
        ],
        local_scheduler=local_scheduler
    )
    print(f"#### output ####")
    print(task.get_output())
