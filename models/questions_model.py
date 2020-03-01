import utils.luigi_wrapper as luigi

from preprocess.data_tokenizer import DataTokenizer
from preprocess.questions_extractor import QuestionsExtractor
from utils.utils import *


class QuestionsModel(luigi.Task):
    def requires(self):
        return {
            'X': DataTokenizer(),
            'y': QuestionsExtractor()
        }

    def run(self):
        X = self.requires()['X'].get_output()
        y = self.requires()['y'].get_output()

        # save_data( ... , self.output().path)
        self.task_done()


if __name__ == '__main__':
    luigi.run_task(QuestionsModel)