import numpy as np

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from utils.utils import *


class DataSet:
    def __init__(self, X, y, train_test_ratio=0.2):
        self.X = csr_matrix(X.values)
        self.y = y.values
        self.train_test_ratio = train_test_ratio
        self.feature_names = X.columns.tolist()

        self._X_cols = X.columns
        self._y_cols = y.columns
        self._index = X.index
        self._arr_indices = np.arange(X.shape[0])
        self.__split_data()

    def __split_data(self):
        args_splitted = train_test_split(self.X, self.y, self._index, self._arr_indices, test_size=self.train_test_ratio)
        self.X_train, self.X_test = args_splitted[0:2]
        self.y_train, self.y_test = args_splitted[2:4]
        self._index_train, self._index_test = args_splitted[4:6]
        self._arr_indices_train, self._arr_indices_test = args_splitted[6:8]

    @property
    def data(self):
        return self.X, self.y

    @property
    def train_data(self):
        return self.X_train, self.y_train

    @property
    def test_data(self):
        return self.X_test, self.y_train

    def get_X(self, as_dataframe: bool = False):
        X = self.X
        if as_dataframe:
            X = pd.DataFrame(X.toarray(), index=self._index, columns=self._X_cols)
        return X

    def get_y(self, as_dataframe: bool = False):
        y = self.y
        if as_dataframe:
            y = pd.DataFrame(y, index=self._index, columns=self._y_cols)
        return y

    def get_X_train(self, as_dataframe: bool = False):
        X_train = self.X_train
        if as_dataframe:
            X_train = pd.DataFrame(X_train.toarray(), index=self._index, columns=self._X_cols)
        return X_train

    def get_y_train(self, as_dataframe: bool = False):
        y_train = self.y_train
        if as_dataframe:
            y_train = pd.DataFrame(y_train, index=self._index_train, columns=self._y_cols)
        return y_train

    def get_X_test(self, as_dataframe: bool = False):
        X_test = self.X_test
        if as_dataframe:
            X_test = pd.DataFrame(X_test.toarray(), index=self._index_test, columns=self._X_cols)
        return X_test

    def get_y_test(self, as_dataframe: bool = False):
        y_test = self.y_test
        if as_dataframe:
            y_test = pd.DataFrame(y_test, index=self._index_test, columns=self._y_cols)
        return y_test
