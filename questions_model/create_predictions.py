import utils.luigi_wrapper as luigi
from preprocess.create_dataset import CreateDataSetTask
from preprocess.dataset import DataSet
from questions_model.choose_best_model import QuestionsModelSelectionTask
from utils.utils import *


class QuestionsMakePredictionsTask(luigi.Task):

    def requires(self):
        return {
            'data': CreateDataSetTask(),
            'best_model': QuestionsModelSelectionTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred.pickle', 'question_model'))

    def run(self):
        inputs = self.get_inputs()
        data: DataSet = inputs['data']
        best_model = inputs['best_model']
        X = data.X
        X_train, y_train = data.train_data

        model = best_model.fit(X_train, y_train)
        y_pred = model.predict_proba(X)

        self.save(y_pred)


if __name__ == '__main__':
    luigi.run_task(QuestionsMakePredictionsTask())
