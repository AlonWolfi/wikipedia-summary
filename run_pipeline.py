from sklearn.metrics import roc_auc_score, f1_score

import utils.luigi_wrapper as luigi
from metrics import SoftMetric, HardMetric
from preprocess.create_dataset import CreateDataSetTask
from prior.belief_prior_predictions import QuestionsBeliefPredictionsAfterPriorTask
from questions_model.create_predictions import QuestionsMakePredictionsTask
from utils.utils import *


class PipelineEnd(luigi.Task):
    def requires(self):
        return {
            'data': CreateDataSetTask(),
            'y_pred': QuestionsMakePredictionsTask(),
            'y_pred_prior_pos': QuestionsBeliefPredictionsAfterPriorTask(is_conn_pos=True,
                                                                         is_after_belief=False),
            'y_pred_prior_neg': QuestionsBeliefPredictionsAfterPriorTask(is_conn_pos=False,
                                                                         is_after_belief=False),
            'y_pred_prior_pos_prior_neg': QuestionsBeliefPredictionsAfterPriorTask(is_conn_pos=False,
                                                                                   is_after_belief=True),
            'y_pred_prior_neg_prior_pos': QuestionsBeliefPredictionsAfterPriorTask(is_conn_pos=True,
                                                                                   is_after_belief=True),
        }

    def output(self):
        output_path = self.config['questions_model']['model_to_use'] + '_experiment' + '.pickle'
        return luigi.LocalTarget(get_file_path(output_path, self.config['exp_dir']))

    def run(self):
        params = {k: v for k, v in self.config.items() if k != 'metric'}
        input = self.get_inputs()
        results = {k: v for k, v in input.items() if k.startswith('y')}

        y_test = input['data'].y_test

        metrics = {
            'roc_auc': SoftMetric(metric=lambda y, yhat: roc_auc_score(y, yhat, average="macro")),
            'f1': HardMetric(metric=lambda y, yhat: f1_score(y, yhat, average="macro"))
        }
        scores = {}
        for m, metric in metrics.items():
            scores[m] = {}
            for k, y_pred in results.items():
                scores[m][k] = metric(y_pred, y_test)

        experiment = {
            'params': params,
            'results': results,
            'scores': scores
        }
        print(experiment)
        save_data(experiment, self.output().path)


if __name__ == '__main__':
    luigi.run_task(PipelineEnd(), local_scheduler=get_from_config('luigi_local_scheduler'), delete_all=False)
