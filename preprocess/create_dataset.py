from sklearn.model_selection import KFold

import utils.luigi_wrapper as luigi
from preprocess.data_tokenization import DataTokenizationTask
from preprocess.dataset import *
from preprocess.feature_selection import FeatureSelectionTask
from preprocess.questions_label_extraction import QuestionsLabelExtractionTask
from utils.utils import *


class _CreateKFoldDataSet(luigi.Task):
    def requires(self):
        return {
            'X': FeatureSelectionTask() if self.config['preprocess']['use_feature_selection']
            else DataTokenizationTask(),
            'y': QuestionsLabelExtractionTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('kfold_dataset.pickle', 'data'))

    def run(self):
        inputs = self.get_inputs()
        X = inputs['X']
        y = inputs['y']
        num_of_folds = int(self.config['preprocess']['num_of_folds'])
        skf = KFold(n_splits=num_of_folds, shuffle=True)
        train_test_splits = [(train_index, test_index) for train_index, test_index in skf.split(X, y)]
        print(train_test_splits)
        self.save({
            'X': X,
            'y': y,
            'train_test_splits': train_test_splits
        })


class CreateDataSetTask(luigi.Task):
    def requires(self):
        return _CreateKFoldDataSet()

    def output(self):
        return luigi.LocalTarget(get_file_path('dataset_' + str(self.config['preprocess']['fold']) + '.pickle', 'data'))

    def run(self):
        dataset = self.get_inputs()
        X = dataset['X']
        y = dataset['y']
        train_test_splits = dataset['train_test_splits']
        train_indices, test_indices = train_test_splits[self.config['preprocess']['fold']]

        print(f'X.shape is {X.shape}')
        print(f'y.shape is {y.shape}')

        ds = FoldDataSet(X, y, train_indices=train_indices, test_indices=test_indices)
        self.save(ds)
