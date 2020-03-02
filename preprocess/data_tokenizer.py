import nltk

import utils.luigi_wrapper as luigi
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

from preprocess.data_extractor import DataExtractor
from utils.utils import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class DataTokenizer(luigi.Task):
    def requires(self):
        return DataExtractor()

    def output(self):
        return luigi.LocalTarget(get_file_path('tokenized_array.pickle'))

    @staticmethod
    def tokenize_doc(doc):
        tokens = nltk.word_tokenize(doc)
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [porter.stem(w) for w in tokens]
        return list(tokens)

    @classmethod
    def __get_vocabulary(cls, texts) -> set:
        vocab: set = set()
        for doc in texts:
            tokens = cls.tokenize_doc(doc)
            vocab = vocab.union(tokens)
        return vocab

    def run(self):
        full_df = self.requires().get_output()

        vocab = self.__get_vocabulary(full_df['text'])

        tokenized_text = full_df['text'].apply(lambda text: ' '.join(self.tokenize_doc(text)))

        vectorizer = TfidfVectorizer()
        vectorizer.fit(tokenized_text)
        transformed_array = vectorizer.transform(tokenized_text)

        # TODO - remove highly-frequent and highly-un-frequent tokens such that
        #  the num of features wouldn't exceed <max_num_of_features> features

        if self.DATAFRAME:
            tokenized_df = pd.DataFrame(index=full_df.index)
            for col, value in zip(vectorizer.get_feature_names(), transformed_array.toarray().T):
                tokenized_df[col] = value
            save_data(tokenized_df, self.output().path)
        else:
            save_data(transformed_array, self.output().path)

if __name__ == '__main__':
    luigi.run_task(DataTokenizer)
