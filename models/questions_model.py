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

from sklearn.base import BaseEstimator

import importlib

import lightgbm as lgb

class TrainerOptimizer:
    def __init__(self, models_dict: dict):
        self.models_dict = models_dict

    @staticmethod
    def load_model(model_path: str) -> BaseEstimator:
        lib_path, model_path = model_path.rsplit(1)
        lib = importlib.import_module(lib_path)
        return getattr(lib, model_path)

    def get_best_model(self):
        pass

class OptunaTrainerOptimizer(TrainerOptimizer):

    @staticmethod
    def optuna_objective(trial, X, y):
        model: BaseEstimator = lgb.sklearn.LGBMClassifier()
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        model.set_params(**params)

        y_pred = cross_val_predict(model, X, y, method='predict_proba')

        score = roc_auc_score(y, y_pred)
        return score

    def get_best_model(self):

        pass


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
        model = OneVsRestClassifier(LogisticRegression())
        self.y_test_pred = cross_val_predict(model, self.X, y_c, cv=3, method='predict_proba')


        save_data(self.y_test_pred, self.output().path)


# General TODO - add prior for questions
if __name__ == '__main__':
    luigi.run_task(QuestionsModel())
