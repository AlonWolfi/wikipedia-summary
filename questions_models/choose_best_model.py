import importlib
import optuna

import utils.luigi_wrapper as luigi

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from questions_models.questions_models import get_models, Model


class TrainerOptimizer:
    def __init__(self, models_dict: dict):
        self.models_dict = models_dict

    @staticmethod
    def load_model(model_path: str) -> BaseEstimator:
        lib_path, model_path = model_path.rsplit('.', 1)
        lib = importlib.import_module(lib_path)
        return getattr(lib, model_path)

    def get_best_model(self):
        pass


class OptunaTrainerOptimizer(TrainerOptimizer):

    def __init__(self, model_dict, X, y):
        super(OptunaTrainerOptimizer, self).__init__(model_dict)
        self.X = X
        self.y = y

    def optuna_objective(self, trial: optuna.Trial):
        model_name = trial.suggest_categorical('model', list(self.models_dict.keys()))

        model_model: Model = self.models_dict[model_name]

        model: BaseEstimator = model_model.model
        params: dict = model_model.params(trial)
        model.set_params(**params)

        model = OneVsRestClassifier(model, n_jobs=-1)

        y_pred = cross_val_predict(model, self.X, self.y, cv=3, method='predict_proba')

        score = roc_auc_score(self.y, y_pred)

        return score

    def get_best_model(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optuna_objective, n_trials=5)
        best_param = study.best_trial.params

        model_name = best_param['model']
        best_clf = get_models()[model_name].model

        del best_param['model']
        best_clf.set_params(best_param)

        return best_clf
