import numpy as np

import utils.luigi_wrapper as luigi
from utils.utils import *

from preprocess.questions_label_extraction import QuestionsLabelExtractionTask
from questions_model.create_predictions import QuestionsMakePredictionsTask
from preprocess.train_test_split import TrainTestSplitTask


class QuestionsPredictionsAfterPriorTask(luigi.Task):
    def requires(self):
        return {'y': QuestionsLabelExtractionTask(),
                'y_pred': QuestionsMakePredictionsTask(),
                'train_test_split': TrainTestSplitTask()}

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred_after_prior.pickle', 'question_model'))

    @staticmethod
    def get_prior(y):
        num_of_pages = y.shape[0]
        num_of_classes = y.shape[1]

        p_ij = np.zeros((num_of_classes, num_of_classes))
        for lst in y:
            for i in range(num_of_classes):
                for j in range(num_of_classes):
                    p_ij[i, j] += (lst[i] * lst[j]) / num_of_pages
        return p_ij

    @staticmethod
    def run_prior_on_prediction(p_ij, prediction):
        sum_prediction = sum(prediction)
        priored_prediction = []
        for i in range(len(prediction)):
            p_i_new = 0
            Z = 0
            for j in range(len(prediction)):
                prior_shift = np.log2(p_ij[i, j] / (p_ij[j, j]) + 1) # * sum_prediction
                p_i_new += prediction[j] * prior_shift
                Z += prior_shift
            p_i_new /= Z

            p_i_new *= prediction[i]

            # p_i_new += prediction[i]
            # p_i_new /= 2

            priored_prediction.append(p_i_new)
        return priored_prediction

    def run(self):
        y_pred = self.get_inputs()['y_pred']
        y = self.get_inputs()['y']
        train_indices = self.get_inputs()['train_test_split']['train_indices']

        p_ij = self.get_prior(y[train_indices])

        y_pred_after_prior = np.array([self.run_prior_on_prediction(p_ij, p) for p in y_pred])

        save_data(y_pred_after_prior, self.output().path)


if __name__ == '__main__':
    luigi.run_task(PriorTask())
