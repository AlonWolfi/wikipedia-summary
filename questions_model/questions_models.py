import importlib

import lightgbm as lgb
import optuna
import sklearn
import xgboost as xgb
from sklearn.base import BaseEstimator


class Model:
    @property
    def model(self) -> BaseEstimator:
        raise NotImplementedError

    def optuna_params(self, trial: optuna.Trial) -> dict:
        raise NotImplementedError

    @staticmethod
    def load_model_from_path(model_path: str) -> BaseEstimator:
        lib_path, model_path = model_path.rsplit('.', 1)
        lib = importlib.import_module(lib_path)
        return getattr(lib, model_path)


class LogisticRegressionModel(Model):
    @property
    def model(self):
        return sklearn.linear_model.LogisticRegression()

    def optuna_params(self, trial):
        param = {
            'penalty': trial.suggest_categorical('penalty', ['none', 'l2']),
            'C': trial.suggest_loguniform('C', 0.001, 10),
            'tol': trial.suggest_uniform('tol', 0.00001, 0.0002),
            'class_weight': 'balanced'
        }
        return param


class SVCModel(Model):
    @property
    def model(self):
        return sklearn.svm.SVC(probability=True)

    def optuna_params(self, trial):
        param = {
            'C': trial.suggest_loguniform('C', 1e-10, 1e10)
        }
        return param


class LGBModel(Model):
    @property
    def model(self):
        return lgb.sklearn.LGBMClassifier()

    def optuna_params(self, trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            # 'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        return param


class XGBModel(Model):
    @property
    def model(self):
        return xgb.sklearn.XGBClassifier()

    def optuna_params(self, trial):
        param = {
            'silent': 1,
            'objective': 'binary:logistic',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
        }
        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        return param


def get_models():
    return {
        'logistic': LogisticRegressionModel(),
        # TODO - SVC is not Working !!!
        # 'svc': SVCModel(),
        # 'lgb': LGBModel(),
        # 'xgb': XGBModel()
    }
