import numpy as np

import utils.luigi_wrapper as luigi
from experimentation.experiment import Experiment
from utils.utils import *


class CreatePriorTask(luigi.Task):

    def requires(self):
        return CreateDataSetTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('p_ij.pickle', 'prior'))

    @staticmethod
    def _get_prior(y):
        num_of_pages = y.shape[0]
        num_of_classes = y.shape[1]

        p_ij = np.zeros((num_of_classes, num_of_classes))
        for lst in y:
            for i in range(num_of_classes):
                for j in range(num_of_classes):
                    p_ij[i, j] += (lst[i] * lst[j]) / num_of_pages
        return p_ij

    def run(self):
        data: DataSet = self.get_inputs()
        y_train = data.y_train

        p_ij = self._get_prior(y_train)
        self.save(p_ij)


if __name__ == '__main__':
    luigi.run_task(CreatePriorTask())
