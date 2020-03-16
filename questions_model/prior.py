import numpy as np

import utils.luigi_wrapper as luigi
from utils.utils import *

from preprocess.questions_label_extraction import QuestionsLabelExtractionTask
from questions_model.create_predictions import QuestionsMakePredictionsTask

class PriorTask(luigi.Task):
    def requires(self):
        return {'y': QuestionsLabelExtractionTask(),
                'y_pred': QuestionsMakePredictionsTask()}

    def output(self):
        return luigi.LocalTarget(get_file_path('prior_result.pickle', 'question_model'))

    @staticmethod
    def run_prior_on_prediction(pij, prediction):
        sum_prediction = sum(prediction)
        priored_prediction = []
        for i in range(len(prediction)):
            p_i_new = 0
            for j in range(len(prediction)):
                p_i_new += pij[i, j] * prediction[j] / (pij[i, i] * pij[j, j] * sum_prediction)
            p_i_new *= prediction[i]
            priored_prediction.append(p_i_new)
        return priored_prediction


    def run(self):
        y_pred = self.requires()['y_pred'].get_outputs()
        y = self.requires()['y'].get_outputs()

        pij = np.zeros((y.shape[1], y.shape[1]))
        num_pages = y.shape[0]
        for lst in y:
            for i in range(len(lst)):
                for j in range(len(lst)):
                    pij[i, j] += (lst[i] * lst[j]) / num_pages

        priored_predictions = [self.run_prior_on_prediction(pij, p) for p in y_pred]

        save_data(np.array(priored_predictions), self.output().path)


if __name__ == '__main__':
    luigi.run_task(PriorTask())
