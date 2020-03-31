import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

import utils.luigi_wrapper as luigi
from preprocess.create_dataset import CreateDataSetTask
from prior.create_entropy import CreateEntropyTask
from prior.create_prior import CreatePriorTask
from questions_model.create_predictions import QuestionsMakePredictionsTask
from utils.utils import *


class BeliefPropagation:

    def __init__(self, T, p_ij, is_negetive_tree=False):
        self.T = T
        self.p_ij = p_ij
        # self.is_negetive_tree = is_negetive_tree
        self.messages_dict = {}

    @staticmethod
    def theta(p_ij, i, j, i_val, j_val):
        if i_val == 1 and j_val == 1:
            return p_ij[i, j] / (p_ij[i, i] * p_ij[j, j])
        elif i_val == 1 and j_val == 0:
            return (p_ij[i, i] - p_ij[i, j]) / (p_ij[i, i] * (1 - p_ij[j, j]))
        elif i_val == 0 and j_val == 1:
            return (p_ij[j, j] - p_ij[i, j]) / (p_ij[j, j] * (1 - p_ij[i, i]))
        elif i_val == 0 and j_val == 0:
            return (1 - p_ij[j, j] - p_ij[i, i] + p_ij[i, j]) / ((1 - p_ij[j, j]) * (1 - p_ij[i, i]))

    def have_common_edge(self, i, j):
        return self.T[i, j] != 0 or self.T[j, i] != 0

    def get_remaining_neighbors(self, i, prediction, parent=None):
        N = []
        for j in range(len(prediction)):
            if self.have_common_edge(i, j) and j != i and j != parent:
                N.append(j)
        return N

    def propagate_message(self, prediction, i, parent, parent_val):
        message = [1., 1.]
        neighbors = self.get_remaining_neighbors(i, prediction, parent)

        if (i, parent, parent_val) in self.messages_dict:
            message = self.messages_dict[(i, parent, parent_val)]
        else:
            for val in range(len(message)):
                message[val] = self.theta(self.p_ij, i, parent, val, parent_val)
                if val == 0:
                    message[val] *= (1 - prediction[i])
                else:
                    message[val] *= prediction[i]
                if len(neighbors) > 0:
                    message[val] *= np.prod([self.propagate_message(prediction, n, i, val) for n in neighbors])
            # cache message
            self.messages_dict[(i, parent, parent_val)] = message
        # print(f'message at {(i, parent, parent_val)} was {message}')
        return np.sum(message)

    def message(self, prediction, i):
        neighbors = self.get_remaining_neighbors(i, prediction)
        message = [1., 1.]
        for val in range(len(message)):
            if val == 0:
                message[val] *= (1 - prediction[i])
            else:
                message[val] *= prediction[i]
            if len(neighbors) > 0:
                message[val] *= np.prod([self.propagate_message(prediction, n, i, val) for n in neighbors])
        return message


class QuestionsBeliefPredictionsAfterPriorTask(luigi.Task):
    is_conn_pos = luigi.luigi.BoolParameter()
    is_after_belief = luigi.luigi.BoolParameter()

    def requires(self):
        req = {
            'data': CreateDataSetTask(),
            'p_ij': CreatePriorTask(),
            'E_ij': CreateEntropyTask()
        }
        if self.is_after_belief:
            print(f'self.is_after_belief: {self.is_after_belief}')
            req['y_pred'] = QuestionsBeliefPredictionsAfterPriorTask(is_conn_pos=(not self.is_conn_pos),
                                                                     is_after_belief=False)
        else:
            req['y_pred'] = QuestionsMakePredictionsTask()

        return req

    def output(self):
        output_path = self.input()['y_pred'].path.split('.')[0]
        if self.is_conn_pos:
            output_path += '_prior_pos'
        else:
            output_path += '_prior_neg'
        output_path += '.pickle'
        return luigi.LocalTarget(get_file_path(output_path, 'question_model'))

    @staticmethod
    def run_prior_on_prediction(p_ij, T, prediction, is_positive_tree=True):
        # sum_prediction = sum(prediction)
        priored_prediction = []
        belief = BeliefPropagation(T=T, p_ij=p_ij)
        for i in range(len(prediction)):
            message = belief.message(prediction, i)
            p_i0, p_i1 = message
            p_i_new = p_i1 / (p_i1 + p_i0)
            # print(f'at i={i} - p_i0, p_i1: {(p_i0, p_i1)}')
            # print(f'at i={i} - p_i_new: {p_i_new}')
            priored_prediction.append(p_i_new)
        return priored_prediction

    @staticmethod
    def get_neg_conn_strength(p_ij):
        conn_strength = np.zeros(p_ij.shape)
        for i in range(p_ij.shape[0]):
            for j in range(p_ij.shape[0]):
                conn_strength[i, j] = (BeliefPropagation.theta(p_ij, i, j, 1, 0) + BeliefPropagation.theta(p_ij, i, j,
                                                                                                           0, 1)) / 2
        return conn_strength

    def run(self):
        inputs = self.get_inputs()
        data = inputs['data']
        y_pred = read_data(self.input()['y_pred'].path)
        p_ij = inputs['p_ij']
        conn_strength_ij = inputs['E_ij']
        neg_conn_strength_ij = self.get_neg_conn_strength(p_ij)

        if self.is_after_belief:
            y_pred = self.config['metric'].get_y(y_pred, data.y_test, normalize=True)

        T_pos = minimum_spanning_tree(conn_strength_ij).todense()
        T_neg = minimum_spanning_tree(neg_conn_strength_ij).todense()
        if self.is_conn_pos:
            y_pred_after_prior = np.array([self.run_prior_on_prediction(p_ij, T_pos, p) for p in y_pred])
        else:
            y_pred_after_prior = np.array([self.run_prior_on_prediction(p_ij, T_neg, p) for p in y_pred])

        save_data(y_pred_after_prior, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsBeliefPredictionsAfterPriorTask())
