import numpy as np
from sklearn.model_selection import train_test_split

from utils.utils import *


class DataSet:
    def __init__(self, X, y, index, train_test_ratio=0.2):
        self.X = X
        self.y = y
        self.index = index
        self.train_test_ratio = train_test_ratio
        self._arr_indices = np.arange(X.shape[0])
        self.__split_data()

    def __split_data(self):
        args_splitted = train_test_split(self.X, self.y, self.index, self._arr_indices, test_size=self.train_test_ratio)
        self.X_train, self.X_test = args_splitted[0:2]
        self.y_train, self.y_test = args_splitted[2:4]
        self.index_train, self.index_test = args_splitted[4:6]
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
            X = pd.DataFrame(X, index=self.index)
        return X

    def get_y(self, as_dataframe: bool = False):
        y = self.y
        if as_dataframe:
            y = pd.DataFrame(y, index=self.index)
        return y

    def get_X_train(self, as_dataframe: bool = False):
        X = self.X
        if as_dataframe:
            X = pd.DataFrame(X, index=self.index_train)
        return X

    def get_y_train(self, as_dataframe: bool = False):
        y = self.y
        if as_dataframe:
            y = pd.DataFrame(y, index=self.index_train)
        return y

    def get_X_test(self, as_dataframe: bool = False):
        X = self.X
        if as_dataframe:
            X = pd.DataFrame(X, index=self.index_test)
        return X

    def get_y_test(self, as_dataframe: bool = False):
        y = self.y
        if as_dataframe:
            y = pd.DataFrame(y, index=self.index_test)
        return y
