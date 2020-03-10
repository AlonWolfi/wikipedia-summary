import numpy as np

from sklearn.metrics import roc_curve, auc
from scipy import interp

import utils.luigi_wrapper as luigi

from preprocess.questions_label_extraction import QuestionsLabelExtractor
from questions_models.create_predictions import QuestionsModel
from utils.utils import *


class PlotROCTask(luigi.Task):
    PLOT_ALL_ROCS = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            'y_true': QuestionsLabelExtractor(),
            'y_pred': QuestionsModel()
        }

    def output(self):
        return luigi.LocalTarget(get_file_path(f"questions_ROC.jpg", 'visualizations'))

    @staticmethod
    def print_metrics(y_true, y_pred):
        # print("Accuracy = ",accuracy_score(y,y_pred))
        # print("f1_micro = ", f1_score(y_true, y_pred, average="micro"))
        # print("f1_macro = ", f1_score(y_true, y_pred, average="macro"))
        # print("f1_weighted = ", f1_score(y_true, y_pred, average="weighted"))
        # print("Hamming loss = ",hamming_loss(y,y_pred))
        pass

    @staticmethod
    def _calculate_singles_ROCs(y_true, y_pred):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for c in range(y_true.shape[1]):
            fpr_values, tpr_values, _ = roc_curve(y_true[:, c], y_pred[:, c])
            auc_score = auc(fpr_values, tpr_values)
            if not np.isnan(auc_score):
                fpr[c], tpr[c] = fpr_values, tpr_values
                roc_auc[c] = auc_score
        return fpr, tpr, roc_auc

    @staticmethod
    def _calculate_macro_ROC(fpr, tpr, roc_auc):
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[c] for c in fpr.keys()]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for c in fpr.keys():
            mean_tpr += interp(all_fpr, fpr[c], tpr[c])

        # Finally average it and compute AUC
        mean_tpr /= len(fpr.keys())

        # Compute ROC of Macro
        fpr_values = all_fpr
        tpr_values = mean_tpr
        roc_auc_values = auc(fpr_values, tpr_values)
        return fpr_values, tpr_values, roc_auc_values

    @staticmethod
    def _calculate_micro_ROC(y_true, y_pred):
        # Compute ROC of Micro
        fpr_values, tpr_values, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc_values = auc(fpr_values, tpr_values)
        return fpr_values, tpr_values, roc_auc_values

    def calculate_ROCs(self):
        fpr, tpr, roc_auc = self._calculate_singles_ROCs(self.y_true, self.y_pred)
        fpr['macro'], tpr['macro'], roc_auc['macro'] = self._calculate_macro_ROC(fpr, tpr, roc_auc)
        fpr['micro'], tpr['micro'], roc_auc['micro'] = self._calculate_micro_ROC(self.y_true, self.y_pred)
        return fpr, tpr, roc_auc

    @staticmethod
    def _plot_agg_ROCs(fpr, tpr, roc_auc):
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 linestyle=':', linewidth=4)

    @staticmethod
    def _plot_all_ROCs(fpr, tpr, roc_auc, lw):
        for c in fpr.keys():
            plt.plot(fpr[c], tpr[c], lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(c, roc_auc[c]))

    @staticmethod
    def _set_ROC_axis(lw, PLOT_ALL_ROCS):
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        if not PLOT_ALL_ROCS:
            plt.legend(loc="lower right")

    def plot_ROCs(self, fpr, tpr, roc_auc, lw=2) -> plt.Figure:
        self._plot_agg_ROCs(fpr, tpr, roc_auc)
        if self.PLOT_ALL_ROCS:
            self._plot_all_ROCs(fpr, tpr, roc_auc, lw)
        self._set_ROC_axis(lw, self.PLOT_ALL_ROCS)
        f = plt.gcf()
        plt.show()
        return f

    def run(self):
        self.y_true = self.requires()['y_true'].get_outputs()
        self.y_pred = self.requires()['y_pred'].get_outputs()

        # metrics
        self.print_metrics(self.y_true, self.y_pred)

        # get ROC auc's
        fpr, tpr, roc_auc = self.calculate_ROCs()

        # get ROC curve plots
        f = self.plot_ROCs(fpr, tpr, roc_auc)

        f.savefig(self.output().path)
        # self.task_done()


# General TODO - add prior for questions
if __name__ == '__main__':
    luigi.run_task(PlotROCTask())
