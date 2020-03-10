import nltk
import numpy as np

import utils.luigi_wrapper as luigi
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

from preprocess.data_tokenization import DataTokenizer
from utils.utils import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureSelectionTask(luigi.Task):
    def requires(self):
        return DataTokenizer()

    def output(self):
        return luigi.LocalTarget(get_file_path('final_data.pickle', 'old__data'))

    def run(self):
        X = self.get_inputs()

        feature_indices = np.random.choice(np.arange(X.shape[1]), size=1000)

        if self.config['preprocess']['is_data_dataframe']:
            X = X.iloc[:, feature_indices]
        else:
            X = X[:, feature_indices]

        save_data(X, self.output().path)


if __name__ == '__main__':
    luigi.run_task(FeatureSelectionTask())
