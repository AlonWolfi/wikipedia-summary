import utils.luigi_wrapper as luigi
from extraction.data_extraction import DataExtractionTask
from preprocess.data_tokenization import DataTokenizationTask
from preprocess.dataset import DataSet
from preprocess.feature_selection import FeatureSelectionTask
from preprocess.questions_label_extraction import QuestionsLabelExtractionTask
from utils.utils import *


class CreateDataSetTask(luigi.Task):
    def requires(self):
        return {
            'X': FeatureSelectionTask() if self.config['preprocess']['use_feature_selection']
            else DataTokenizationTask(),
            'y': QuestionsLabelExtractionTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('dataset.pickle', 'data'))

    def run(self):
        inputs = self.get_inputs()
        X = inputs['X']
        y = inputs['y']
        train_test_ratio = self.config['preprocess']['train_test_ratio']

        print(f'X.shape is {X.shape}')
        print(f'y.shape is {y.shape}')
        print(f'train_test_ratio is {train_test_ratio}')

        ds = DataSet(X, y, train_test_ratio=train_test_ratio)
        self.save(ds)
