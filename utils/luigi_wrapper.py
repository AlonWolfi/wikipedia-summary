import luigi
from luigi import LocalTarget
from luigi import Parameter, BoolParameter
from luigi import build
import shutil

from utils.utils import *


class Task(luigi.Task):

    def __init__(self):
        super(Task, self).__init__()
        self.config = get_config()

    @classmethod
    def __get_task_done_path(cls):
        return get_project_dir() / 'cache' / (cls.__name__ + '.done')

    def output(self):
        return LocalTarget(self.__get_task_done_path())

    @classmethod
    def get_outputs(cls):
        ins = cls()
        return loop_through_iterable(ins.output(), lambda _output: read_data(_output.path))

    def get_inputs(self):
        return loop_through_iterable(self.input(), lambda _input: read_data(_input.path))

    @classmethod
    def task_done(cls):
        empty_file = ''
        file_path = cls.__get_task_done_path()
        save_data(empty_file, file_path)

    def save(self, data):
        return save_data(data, self.output().path)

    def run(self):
        self.task_done()


def run_task(task: Task, local_scheduler: bool = False, delete_all: bool = False):
    if get_from_config('delete_cache', 'debug'):
        if delete_all:
            shutil.rmtree(get_project_dir() / 'cache')
        else:
            def remove_cache(output_path):
                validate_path(output_path)
                if os.path.exists(output_path):
                    os.remove(output_path)

            loop_through_iterable(task.output(), remove_cache)

    luigi.build(
        [
            task
        ],
        local_scheduler=local_scheduler
    )
    print(f"#### output ####")
    print(task.get_outputs())
