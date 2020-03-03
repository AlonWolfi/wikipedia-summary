import utils.luigi_wrapper as luigi
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

from preprocess.data_extractor import DataExtractor
from preprocess.page_list_extractor_xml import PageListExtractorTask
from utils.utils import *


class QuestionsExtractor(luigi.Task):
    NOT_FREQ_LABEL_THRESH = get_from_config('NOT_FREQ_LABEL_THRESH')

    def requires(self):
        return DataExtractor()

    def output(self):
        return luigi.LocalTarget(get_file_path('questions.pickle'))

    @staticmethod
    def __get_questions_from_infobox(infobox):
        return list(infobox.keys())

    @classmethod
    def __filter_small_classes(cls, questions):
        labels = {}
        for q in questions:
            for label in q:
                if label in labels.keys():
                    labels[label] += 1
                else:
                    labels[label] = 1

        labels = [key for key, value in labels.items() if value > cls.NOT_FREQ_LABEL_THRESH]

        filtered_questions = []
        for q in questions:
            filtered_q = []
            for label in q:
                if label in labels:
                    filtered_q.append(label)
            filtered_questions.append(filtered_q)

        return filtered_questions

    def run(self):
        full_df = self.requires().get_output()

        questions = full_df['infobox'].apply(self.__get_questions_from_infobox)
        filtered_questions = self.__filter_small_classes(questions)

        vectorizer = MultiLabelBinarizer()
        vectorizer.fit(filtered_questions)
        transformed_array = vectorizer.transform(filtered_questions)

        if self.DATAFRAME:
            df_questions = pd.DataFrame(index=full_df.index)
            for col, value in zip(vectorizer.classes_, transformed_array.T):
                df_questions[col] = value
            save_data(df_questions, self.output().path)
        else:
            save_data(transformed_array, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsExtractor)
