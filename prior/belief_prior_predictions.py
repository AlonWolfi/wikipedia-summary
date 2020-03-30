import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

import utils.luigi_wrapper as luigi
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

    def theta(self, i, j, i_val, j_val):
        p_ij = self.p_ij
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
                message[val] = self.theta(i, parent, val, parent_val)
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
    def requires(self):
        return {
            'y_pred': QuestionsMakePredictionsTask(),
            'p_ij': CreatePriorTask(),
            'E_ij': CreateEntropyTask()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path('y_pred_after_prior.pickle', 'question_model'))

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

    def run(self):
        inputs = self.get_inputs()
        y_pred = inputs['y_pred']
        p_ij = inputs['p_ij']
        E_ij = inputs['E_ij']
        T_pos = minimum_spanning_tree(E_ij).todense()

        y_pred_after_prior = np.array([self.run_prior_on_prediction(p_ij, T_pos, p) for p in y_pred])

        save_data(y_pred_after_prior, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsBeliefPredictionsAfterPriorTask())
