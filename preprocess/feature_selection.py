import numpy as np

import utils.luigi_wrapper as luigi
from preprocess.data_tokenization import DataTokenizationTask
from utils.utils import *


class FeatureSelectionTask(luigi.Task):
    def requires(self):
        return DataTokenizationTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('final_data.pickle', 'data'))

    def run(self):
        X = self.get_task_inputs()

        feature_indices = np.random.choice(np.arange(X.shape[1]), size=1000)

        X = X[:, feature_indices]

        save_data(X, self.output().path)


if __name__ == '__main__':
    luigi.run_task(FeatureSelectionTask())
