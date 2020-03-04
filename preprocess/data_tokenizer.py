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
    NOT_FREQ_TOKEN_THRESH = get_from_config('NOT_FREQ_TOKEN_THRESH')

    def requires(self):
        return DataExtractor()

    def output(self):
        return luigi.LocalTarget(get_file_path('tokenized_array.pickle', 'data'))

    @staticmethod
    def tokenize_doc(doc):
        tokens = nltk.word_tokenize(doc)
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [porter.stem(w) for w in tokens]
        return list(tokens)

    @classmethod
    def __get_vocabulary(cls, texts) -> set:
        vocab: dict = dict()
        for doc in texts:
            tokens = set(cls.tokenize_doc(doc))
            for t in tokens:
                if t in vocab.keys():
                    vocab[t] += 1
                else:
                    vocab[t] = 1
        vocab = [key for key, value in vocab.items() if value > cls.NOT_FREQ_TOKEN_THRESH * len(texts)]
            
        return vocab
           

    def run(self):
        full_df = self.input()

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
