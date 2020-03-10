from sklearn.model_selection import cross_val_predict

import utils.luigi_wrapper as luigi

from utils.utils import *

from preprocess.feature_selection import FeatureSelectionTask
from preprocess.questions_label_extraction import QuestionsLabelExtractionTask
from questions_model.choose_best_model import QuestionsModelSelectionTask


class QuestionsMakePredictionsTask(luigi.Task):

    def requires(self):
        return {
            'X': FeatureSelectionTask(),
            'y': QuestionsLabelExtractionTask(),
            'best_model': QuestionsModelSelectionTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred.pickle', 'question_model'))

    def run(self):
        X = self.requires()['X'].get_outputs()
        y = self.requires()['y'].get_outputs()
        best_model = self.requires()['best_model'].get_outputs()

        if self.config['preprocess']['is_data_dataframe']:
            X = X.to_numpy()
            y = y.to_numpy()

        y_pred = cross_val_predict(best_model, X, y, method='predict_proba')

        save_data(y_pred, self.output().path)


# General TODO - add prior for questions
if __name__ == '__main__':
    luigi.run_task(QuestionsMakePredictionsTask())
