from sklearn.model_selection import cross_val_predict

import utils.luigi_wrapper as luigi

from utils.utils import *

from preprocess.feature_selection import FeatureSelectionTask
from preprocess.questions_label_extraction import QuestionsLabelExtractionTask
from questions_model.choose_best_model import QuestionsModelSelectionTask
from preprocess.train_test_split import TrainTestSplitTask


class QuestionsMakePredictionsTask(luigi.Task):

    def requires(self):
        return {
            'X': FeatureSelectionTask(),
            'y': QuestionsLabelExtractionTask(),
            'best_model': QuestionsModelSelectionTask(),
            'train_test_split': TrainTestSplitTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred.pickle', 'question_model'))

    def run(self):
        X = self.requires()['X'].get_outputs()
        y = self.requires()['y'].get_outputs()
        best_model = self.requires()['best_model'].get_outputs()
        train_indexes = self.requires()['train_test_split'].get_outputs()['train_indexes']

        if self.config['preprocess']['is_data_dataframe']:
            X = X.to_numpy()
            y = y.to_numpy()

        x_train = X[train_indexes]
        y_train = y[train_indexes]

        model = best_model.fit(x_train, y_train)
        y_pred = model.predict_proba(X)

        save_data(y_pred, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsMakePredictionsTask())
