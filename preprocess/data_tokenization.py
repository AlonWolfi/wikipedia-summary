import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import utils.luigi_wrapper as luigi
from utils.utils import *

from extraction.data_extraction import DataExtractionTask


class DataTokenizationTask(luigi.Task):
    NOT_FREQ_TOKEN_THRESH = get_from_config('NOT_FREQ_TOKEN_THRESH', 'preprocess')

    def requires(self):
        return DataExtractionTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('tokenized_array.pickle', 'data'))

    @staticmethod
    def tokenize_doc(doc, vocab=None):
        tokens = nltk.word_tokenize(doc)
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [porter.stem(w) for w in tokens]
        if vocab:
            tokens = [w for w in tokens if w in vocab]
        return list(tokens)

    @classmethod
    def __get_vocabulary(cls, texts) -> set:
        words_dict = dict()
        for doc in texts:
            tokens = set(cls.tokenize_doc(doc))
            for t in tokens:
                if t in words_dict.keys():
                    words_dict[t] += 1
                else:
                    words_dict[t] = 1
        vocab: set = set([key for key, value in words_dict.items() if value > cls.NOT_FREQ_TOKEN_THRESH * len(texts)])

        return vocab

    @classmethod
    def get_vocabulary(cls, texts) -> set:
        words_dict = dict()
        for doc in texts:
            tokens = set(cls.tokenize_doc(doc))
            for t in tokens:
                if t in words_dict.keys():
                    words_dict[t] += 1
                else:
                    words_dict[t] = 1
        vocab: set = set([key for key, value in words_dict.items() if value > cls.NOT_FREQ_TOKEN_THRESH * len(texts)])

        return vocab

    def run(self):
        full_df = self.get_task_inputs()

        vocab = self.__get_vocabulary(full_df['text'])

        tokenized_text = full_df['text'].apply(lambda text: ' '.join(self.tokenize_doc(text, vocab=vocab)))

        vectorizer = TfidfVectorizer()
        vectorizer.fit(tokenized_text)
        transformed_array = vectorizer.transform(tokenized_text)

        tokenized_df = pd.DataFrame(transformed_array.toarray(), index=full_df.index,
                                    columns=vectorizer.get_feature_names())
        save_data(tokenized_df, self.output().path)

        # if self.config['preprocess']['is_data_dataframe']:
        #     tokenized_df = pd.DataFrame(index=full_df.index)
        #     for col, value in zip(vectorizer.get_feature_names(), transformed_array.toarray().T):
        #         tokenized_df[col] = value
        #     save_data(tokenized_df, self.output().path)
        # else:
        #     save_data(transformed_array, self.output().path)


if __name__ == '__main__':
    luigi.run_task(DataTokenizationTask())
