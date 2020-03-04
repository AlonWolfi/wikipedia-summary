import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss, precision_score, precision_recall_curve
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LinearRegression, LogisticRegression

import utils.luigi_wrapper as luigi

from preprocess.data_tokenizer import DataTokenizer
from preprocess.questions_label_extractor import QuestionsLabelExtractor
from sklearn.multiclass import OneVsRestClassifier
from utils.utils import *


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

        # Preprocess
        # X = self.__random_features(X)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
        #
        #                                                                         # fit & pred
        #         model.fit(self.X_train, self.y_train)
        #
        #         self.y_test_pred = model.predict(self.X_test)shuffle=True)
        # Choose model
        # TODO - try different models and add grid search
        model = LogisticRegression()

        self.y_test_pred = None

        for c in range(self.y.shape[1]):
            y_c = self.y[:, c]
            y_c_pred = cross_val_predict(model, self.X, y_c, cv=2).reshape((-1, 1))
            if self.y_test_pred is not None:
                self.y_test_pred = np.append(self.y_test_pred, y_c_pred, axis=1)
            else:
                self.y_test_pred = y_c_pred

        save_data(self.y_test_pred, self.output().path)


# General TODO - add prior for questions
if __name__ == '__main__':
    luigi.run_task(QuestionsModel())
