from sklearn.model_selection import cross_val_predict

import utils.luigi_wrapper as luigi

from preprocess.feature_selection import FeatureSelectionTask
from extraction.questions_label_extractor import QuestionsLabelExtractor

from utils.utils import *

from questions_models.choose_best_model import ModelSelectionTask


class QuestionsModel(luigi.Task):

    def requires(self):
        return {
            'X': FeatureSelectionTask(),
            'y': QuestionsLabelExtractor(),
            'best_model': ModelSelectionTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred.pickle', 'question_model'))

    def run(self):
        self.X = self.requires()['X'].get_outputs()
        self.y = self.requires()['y'].get_outputs()
        self.best_model = self.requires()['best_model'].get_outputs()

        if self.DATAFRAME:
            self.X = self.X.to_numpy()
            self.y = self.y.to_numpy()

        self.y_test_pred = cross_val_predict(self.best_model, self.X, self.y, method='predict_proba')

        save_data(self.y_test_pred, self.output().path)


# General TODO - add prior for questions
if __name__ == '__main__':
    luigi.run_task(QuestionsModel())
