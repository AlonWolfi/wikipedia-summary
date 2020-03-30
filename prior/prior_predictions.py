import numpy as np

import utils.luigi_wrapper as luigi
from prior.create_entropy import CreateEntropyTask
from prior.create_prior import CreatePriorTask
from questions_model.create_predictions import QuestionsMakePredictionsTask
from utils.utils import *


class QuestionsPredictionsAfterPriorTask(luigi.Task):
    def requires(self):
        return {
            'y_pred': QuestionsMakePredictionsTask(),
            'p_ij': CreatePriorTask(),
            'E_ij': CreateEntropyTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred_after_prior_old.pickle', 'question_model'))

    @staticmethod
    def _run_prior_on_prediction(p_ij, E_ij, prediction):
        sum_prediction = sum(prediction)
        priored_prediction = []
        for i in range(len(prediction)):
            p_i_new = 0
            Z = 0
            for j in range(len(prediction)):
                # prior_shift = E_ij[i, j]  # * sum_prediction

                mone = (prediction[j] * p_ij[i, j] + (1 - prediction[j]) * (p_ij[i, i] - p_ij[i, j]))
                mechane = mone + (1 - prediction[i]) * (
                        prediction[j] * (p_ij[j, j] - p_ij[i, j]) + (1 - prediction[j]) * (
                        1 - p_ij[i, i] - p_ij[i, j] + p_ij[i, j]))

                p_i_new += prediction[i] * (mone / mechane)
                Z += (mone / mechane)

            p_i_new /= Z

            priored_prediction.append(p_i_new)
        return priored_prediction

    def run(self):
        inputs = self.get_inputs()
        y_pred = inputs['y_pred']
        p_ij = inputs['p_ij']
        E_ij = inputs['E_ij']

        y_pred_after_prior = np.array([self._run_prior_on_prediction(p_ij, E_ij, p) for p in y_pred])

        save_data(y_pred_after_prior, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsPredictionsAfterPriorTask())
