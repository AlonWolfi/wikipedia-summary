import luigi
from luigi import LocalTarget
from luigi import Parameter, BoolParameter
from luigi import build

from utils.utils import read_data


class Task(luigi.Task):
    DEBUG = luigi.BoolParameter

    @classmethod
    def get_output(cls):
        ins = cls()
        output_path = ins.output().path
        print(f'{cls} output path is {output_path}')
        return read_data(output_path)

