from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import utils.luigi_wrapper as luigi
from extraction.data_extraction import DataExtractionTask
from utils.utils import *

porter = PorterStemmer()


class QuestionsLabelExtractionTask(luigi.Task):
    NOT_FREQ_LABEL_THRESH = get_from_config('NOT_FREQ_LABEL_THRESH', 'preprocess')

    def requires(self):
        return DataExtractionTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('question_labels.pickle', 'data'))

    @staticmethod
    def __get_questions_from_infobox(infobox):
        questions_cleaned = [porter.stem(w) for w in list(infobox.keys())]
        return questions_cleaned

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
        full_df = self.requires().get_task_outputs()

        questions = full_df['infobox'].apply(self.__get_questions_from_infobox)
        filtered_questions = self.__filter_small_classes(questions)

        vectorizer = MultiLabelBinarizer()
        vectorizer.fit(filtered_questions)
        transformed_array = vectorizer.transform(filtered_questions)

        df_questions = pd.DataFrame(transformed_array, index=full_df.index, columns=vectorizer.classes_)
        save_data(df_questions, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsLabelExtractionTask())
