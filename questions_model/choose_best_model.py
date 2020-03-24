import importlib

import optuna
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

import utils.luigi_wrapper as luigi
from preprocess.create_dataset import CreateDataSetTask
from preprocess.dataset import DataSet
from questions_model.questions_models_config import get_models, OptunaModel
from utils.utils import *


class ParamOptimizer:
    def __init__(self, models_dict: dict):
        self.models_dict = models_dict

    @staticmethod
    def load_model_from_path(model_path: str) -> BaseEstimator:
        lib_path, model_path = model_path.rsplit('.', 1)
        lib = importlib.import_module(lib_path)
        return getattr(lib, model_path)

    def get_best_model(self):
        pass


class OptunaParamOptimizer(ParamOptimizer):

    def __init__(self, model_dict, X, y, one_vs_rest):
        super(OptunaParamOptimizer, self).__init__(model_dict)
        self.X = X
        self.y = y
        self.one_vs_rest = one_vs_rest

    def optuna_objective(self, trial: optuna.Trial):
        model_name = trial.suggest_categorical('model', list(self.models_dict.keys()))

        model_model: OptunaModel = self.models_dict[model_name]

        model: BaseEstimator = model_model.model
        params: dict = model_model.params(trial)
        model.set_params(**params)

        if self.one_vs_rest:
            model = OneVsRestClassifier(model, n_jobs=-1)

        y_pred = cross_val_predict(model, self.X, self.y, cv=3)

        score = f1_score(self.y, y_pred, average='macro')

        return score

    def get_best_model(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optuna_objective, n_trials=5, n_jobs=-1)
        best_param = study.best_trial.params

        model_name = best_param['model']
        best_clf = get_models()[model_name].model

        del best_param['model']

        best_clf.set_params(**best_param)

        if self.one_vs_rest:
            best_clf = OneVsRestClassifier(best_clf, n_jobs=-1)

        return best_clf


class QuestionsModelSelectionTask(luigi.Task):
    def requires(self):
        return CreateDataSetTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('best_estimator.pickle', 'question_model'))

    def run(self):
        data: DataSet = self.get_inputs()
        X_train, y_train = data.train_data

        param_optimizer = OptunaParamOptimizer(get_models(), X_train, y_train,
                                               self.config['questions_model']['one_vs_rest'])

        best_clf = param_optimizer.get_best_model()

        self.save(best_clf)
