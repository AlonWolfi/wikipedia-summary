import shutil

import luigi
import winsound
from luigi import LocalTarget
from sklearn.metrics import f1_score

from metrics import HardMetric
from utils.utils import *


class Task(luigi.Task):

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.config = get_config()
        self.config['metric'] = HardMetric(metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'))
        self.config['exp_dir'] = ['questions_model', self.config['questions_model']['model_to_use'],
                                  self.config['preprocess']['fold']]

    @classmethod
    def __get_task_done_path(cls):
        return get_project_dir() / 'cache' / (cls.__name__ + '.done')

    def output(self):
        return LocalTarget(self.__get_task_done_path())

    @classmethod
    def get_outputs(cls):
        ins = cls()
        return loop_through_iterable(ins.output(), lambda _output: read_data(_output.path))

    @classmethod
    def get_inputs(cls):
        ins = cls()
        return loop_through_iterable(ins.input(), lambda _input: read_data(_input.path))

    @classmethod
    def task_done(cls):
        empty_file = ''
        file_path = cls.__get_task_done_path()
        save_data(empty_file, file_path)

    def save(self, data):
        # TODO - make work for more than one file (now only works if output is of LocalTarget type)
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
    frequency = 2400  # Set Frequency To 2500 Hertz
    duration = 5 * 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
