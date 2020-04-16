import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

import utils.luigi_wrapper as luigi
from preprocess.create_dataset import CreateDataSetTask
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

    @staticmethod
    def theta_mone(p_ij, i, j, i_val, j_val):
        if i_val == 1 and j_val == 1:
            return p_ij[i, j]
        elif i_val == 1 and j_val == 0:
            return (p_ij[i, i] - p_ij[i, j])
        elif i_val == 0 and j_val == 1:
            return (p_ij[j, j] - p_ij[i, j])
        elif i_val == 0 and j_val == 0:
            return (1 - p_ij[j, j] - p_ij[i, i] + p_ij[i, j])

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
    is_after_belief = luigi.luigi.BoolParameter()
    tree_type = luigi.luigi.Parameter()
    is_normalized = luigi.luigi.BoolParameter(default=False)
    perv_tree_type = luigi.luigi.Parameter(default=' ')

    def requires(self):
        req = {
            'data': CreateDataSetTask()
        }
        if self.is_after_belief:
            req['y_pred'] = QuestionsBeliefPredictionsAfterPriorTask(is_after_belief=False,
                                                                     tree_type=self.perv_tree_type)
        else:
            req['y_pred'] = QuestionsMakePredictionsTask()

        return req

    def output(self):
        output_path = self.input()['y_pred'].path.split('.')[0]
        if self.is_after_belief and self.is_normalized:
            output_path += '_normed'
        output_path += '_prior'
        output_path += '_' + str(self.tree_type)
        output_path += '.pickle'
        return luigi.LocalTarget(get_file_path(output_path, self.config['exp_dir']))

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
    def get_MI_strength(p_ij):
        conn_strength = np.zeros(p_ij.shape)
        for i in range(p_ij.shape[0]):
            for j in range(p_ij.shape[0]):
                conn_strength[i, j] = 0
                for i_val in range(2):
                    for j_val in range(2):
                        conn_addition = BeliefPropagation.theta_mone(p_ij, i, j, i_val, j_val) * np.log(
                            BeliefPropagation.theta(p_ij, i, j, i_val, j_val))
                        if np.isnan(conn_addition):
                            conn_addition = 0
                        elif np.isinf(-conn_addition) or np.isinf(conn_addition):
                            conn_addition = -10
                        conn_strength[i, j] += conn_addition
        return (-1.) * conn_strength

    @staticmethod
    def get_pos_strength(p_ij):
        conn_strength = np.zeros(p_ij.shape)
        for i in range(p_ij.shape[0]):
            for j in range(p_ij.shape[0]):
                conn_addition = BeliefPropagation.theta_mone(p_ij, i, j, 1, 1) * np.log(
                    BeliefPropagation.theta(p_ij, i, j, 1, 1))
                if np.isnan(conn_addition):
                    conn_addition = 0
                elif np.isinf(-conn_addition) or np.isinf(conn_addition):
                    conn_addition = -10
                conn_strength[i, j] = conn_addition
        return (-1.) * conn_strength

    @staticmethod
    def get_semi_pos_strength(p_ij):
        conn_strength = np.zeros(p_ij.shape)
        for i in range(p_ij.shape[0]):
            for j in range(p_ij.shape[0]):
                conn_strength[i, j] = 0
                for i_val in range(2):
                    for j_val in range(2):
                        if i_val != 0 or j_val != 0:
                            conn_addition = BeliefPropagation.theta_mone(p_ij, i, j, i_val, j_val) * np.log(
                                BeliefPropagation.theta(p_ij, i, j, i_val, j_val))
                            if np.isnan(conn_addition):
                                conn_addition = 0
                            elif np.isinf(-conn_addition) or np.isinf(conn_addition):
                                conn_addition = -10
                            conn_strength[i, j] += conn_addition
        return (-1.) * conn_strength

    @staticmethod
    def get_random_strength(p_ij):
        conn_strength = np.random.randn(*p_ij.shape)
        return conn_strength

    @staticmethod
    def _get_prior(y):
        num_of_pages = y.shape[0]
        num_of_classes = y.shape[1]
        p_ij = np.zeros((num_of_classes, num_of_classes))
        for lst in y:
            for i in range(num_of_classes):
                for j in range(num_of_classes):
                    p_ij[i, j] += (lst[i] * lst[j]) / num_of_pages
        return p_ij

    def run(self):
        inputs = self.get_inputs()
        data = inputs['data']
        y_pred = read_data(self.input()['y_pred'].path)
        p_ij = self._get_prior(data.y_train)

        if self.tree_type == 'MI':
            conn_strength_ij = self.get_MI_strength(p_ij)
        elif self.tree_type == 'pos':
            conn_strength_ij = self.get_pos_strength(p_ij)
        elif self.tree_type == 'semi_pos':
            conn_strength_ij = self.get_semi_pos_strength(p_ij)
        elif self.tree_type == 'random':
            conn_strength_ij = self.get_random_strength(p_ij)
        else:
            raise RuntimeError(f'No known tree type: {self.tree_type}')

        if self.is_after_belief and self.is_normalized:
            y_pred = self.config['metric'].get_y(y_pred, data.y_test, normalize=True)

        T_pos = minimum_spanning_tree(conn_strength_ij).todense()
        y_pred_after_prior = np.array([self.run_prior_on_prediction(p_ij, T_pos, p) for p in y_pred])

        save_data(y_pred_after_prior, self.output().path)


if __name__ == '__main__':
    luigi.run_task(QuestionsBeliefPredictionsAfterPriorTask())
