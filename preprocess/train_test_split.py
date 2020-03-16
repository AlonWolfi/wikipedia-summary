import numpy as np

import utils.luigi_wrapper as luigi
from utils.utils import *

from extraction.data_extraction import DataExtractionTask
from sklearn.model_selection import train_test_split


class TrainTestSplitTask(luigi.Task):
    TEST_SIZE = get_from_config('TEST_SIZE', 'preprocess')

    def requires(self):
        return DataExtractionTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('train_test_dict.pickle', 'old__data'))

    def run(self):
        train_test_dict = dict()
        full_df = self.requires().get_outputs()

        train_indexes, test_indexes = train_test_split(list(range(len(full_df))), test_size=self.TEST_SIZE)
        train_test_dict['train_indexes'] = train_indexes
        train_test_dict['test_indexes'] = test_indexes
        train_test_dict['train_pages'] = list(full_df.iloc[train_indexes].index)
        train_test_dict['test_pages'] = list(full_df.iloc[test_indexes].index)

        save_data(train_test_dict, self.output().path)


if __name__ == '__main__':
    luigi.run_task(TrainTestSplitTask())
