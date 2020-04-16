import numpy as np

import utils.luigi_wrapper as luigi
from preprocess.create_dataset import CreateDataSetTask
from preprocess.dataset import DataSet
from questions_model.create_predictions import QuestionsMakePredictionsTask
from utils.utils import *


class QuestionsPredictionsAfterPriorTask(luigi.Task):
    def requires(self):
        return {'data': CreateDataSetTask(),
                'y_pred': QuestionsMakePredictionsTask()}

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred_after_prior.pickle', self.config['exp_dir']))

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
        prior_shift = 0
        all_counter = 0
        one_counter = 0
        for i in range(len(prediction)):
            p_i_new = 0
            Z = 0
            for j in range(len(prediction)):
                if j == i:
                    continue
                if prediction[j] > 0.5:
                    prior_shift = np.ceil(p_ij[i, j])
                if prior_shift == 1:
                    one_counter += 1
                    break

            all_counter += 1

            # prior_shift = np.log2(p_ij[i, j] / (p_ij[j, j]) + 1) # * sum_prediction
            # p_i_new += prediction[j] * prior_shift
            # Z += prior_shift
            # normalization
            # p_i_new /= Z
            p_i_new = prior_shift * prediction[i]

            # multiply by first prediction
            # p_i_new *= prediction[i]

            # p_i_new += prediction[i]
            # p_i_new /= 2

            priored_prediction.append(p_i_new)

        print(f'one_counter: {one_counter}')
        print(f'all counter: {all_counter}')

        return priored_prediction

    def run(self):
        inputs = self.get_task_inputs()
        data: DataSet = inputs['data']
        y_pred = inputs['y_pred']

        p_ij = self.get_prior(data.y_train)

        y_pred_after_prior = np.array([self.run_prior_on_prediction(p_ij, p) for p in y_pred])

        save_data(y_pred_after_prior, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsPredictionsAfterPriorTask())
