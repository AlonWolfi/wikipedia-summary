import numpy as np

import utils.luigi_wrapper as luigi
from prior.create_prior import CreatePriorTask
from utils.utils import *


class CreateEntropyTask(luigi.Task):

    def requires(self):
        return CreatePriorTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('E_ij.pickle', 'prior'))

    @staticmethod
    def _get_mutual_information(p_ij, neginf, nan=0):
        E = np.zeros(p_ij.shape)
        for i in range(p_ij.shape[0]):
            for j in range(p_ij.shape[1]):
                E[i, j] = p_ij[i, j] * np.log(p_ij[i, j] / (p_ij[i, i] * p_ij[j, j]))
        E_filled = (-1.) * np.nan_to_num(E, nan=nan, neginf=neginf)
        return E_filled

    def run(self):
        p_ij = self.get_inputs()

        E_ij = self._get_mutual_information(p_ij, neginf=self.config['prior']['E_neginf_replace'])
        self.save(E_ij)


if __name__ == '__main__':
    luigi.run_task(CreateEntropyTask())
