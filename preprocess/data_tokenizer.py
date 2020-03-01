import os
import glob
import nltk
import gensim
import pandas as pd
import wikipedia
import wikipediaapi
import wptools

import utils.luigi_wrapper as luigi
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

from preprocess.data_extractor import DataExtractor
from preprocess.page_list_extractor import PageListExtractorTask
from utils.utils import *


class DataTokenizer(luigi.Task):
    def requires(self):
        return DataExtractor()

    def output(self):
        return luigi.LocalTarget(get_file_path_from_config('tokenized_data'))

    @staticmethod
    def tokenize_doc(doc) -> set:
        tokens = nltk.word_tokenize(doc)
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [porter.stem(w) for w in tokens]
        return set(tokens)

    def __get_vocabulary(self) -> set:
        vocab: set = set()
        for doc in self.full_df['text']:
            tokens = self.tokenize_doc(doc)
            vocab = vocab.union(tokens)
        return vocab

    def run(self):
        self.full_df = self.requires().get_output()

        vocab = self.__get_vocabulary()

        tokenized_df = self.full_df.copy()

        #### replace for tokenization
        tokenized_df['a'] = 1
        tokenized_df = tokenized_df[['a']]

        save_data(tokenized_df, self.output().path)


if __name__ == '__main__':
    luigi.build(
        [
            DataTokenizer(
                DEBUG=True
            )
        ],
        local_scheduler=False
    )
    print('#### output ####')
    print(DataTokenizer.get_output())
