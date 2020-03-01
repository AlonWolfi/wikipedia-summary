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
from sklearn.preprocessing import MultiLabelBinarizer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

from preprocess.data_extractor import DataExtractor
from preprocess.page_list_extractor import PageListExtractorTask
from utils.utils import *


class QuestionsExtractor(luigi.Task):
    def requires(self):
        return DataExtractor()

    def output(self):
        return luigi.LocalTarget(get_file_path('questions.pickle'))

    @staticmethod
    def __get_questions_from_infobox(infobox):
        return list(infobox.keys())

    def run(self):
        full_df = self.requires().get_output()

        questions = full_df['infobox'].apply(self.__get_questions_from_infobox)

        vectorizer = MultiLabelBinarizer()
        vectorizer.fit(questions)
        transformed_array = vectorizer.transform(questions)

        if self.DATAFRAME:
            df_questions = pd.DataFrame(index=full_df.index)
            for col, value in zip(vectorizer.classes_, transformed_array.T):
                df_questions[col] = value
            save_data(df_questions, self.output().path)
        else:
            save_data(transformed_array, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsExtractor)
