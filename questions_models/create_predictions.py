import numpy as np

from sklearn.model_selection import train_test_split, cross_val_predict

import utils.luigi_wrapper as luigi

from preprocess.data_tokenizer import DataTokenizer
from preprocess.questions_label_extractor import QuestionsLabelExtractor

from utils.utils import *

from questions_models.questions_models import get_models
from questions_models.choose_best_model import OptunaTrainerOptimizer


class QuestionsModel(luigi.Task):

    def requires(self):
        return {
            'X': DataTokenizer(),
            'y': QuestionsLabelExtractor()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred.pickle', 'predictions'))

    @staticmethod
    def __random_features(X, random_features_size=1000):
        return X.iloc[:, list(np.random.randint(X.shape[1], size=random_features_size))]

    def run(self):
        self.X = self.requires()['X'].load_outputs()
        self.y = self.requires()['y'].load_outputs()

        if self.DATAFRAME:
            self.X = self.X.to_numpy()
            self.y = self.y.to_numpy()

        print(f'X.shape is {self.X.shape}')
        print(f'y.shape is {self.y.shape}')

        # Choose model
        # model = OneVsRestClassifier(LogisticRegression())
        # self.y_test_pred = cross_val_predict(model, self.X, self.y, cv=3, method='predict_proba')

        trainer = OptunaTrainerOptimizer(get_models(), self.X, self.y)

        best_clf = trainer.get_best_model()

        self.y_test_pred = cross_val_predict(best_clf, self.X, self.y, method='predict_proba')

        save_data(self.y_test_pred, self.output().path)


# General TODO - add prior for questions
if __name__ == '__main__':
    luigi.run_task(QuestionsModel())
