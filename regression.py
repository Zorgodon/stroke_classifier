#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# cv = tscv, kf, etc
def load(fname: str, training_ratio: float, cv: str, num_splits: int = 5) -> tuple:
    # basic spliter for now
    """Loads 2 column data from a CSV file with 2 columns, x and y"""
    # read data
    data = pd.read_csv(fname)
    y_column = 'stroke'
    # get training data
    n_rows = int(len(data) * training_ratio)

    X_columns = list(data.columns)
    X_columns.remove(y_column)
    X = data[X_columns].to_numpy()
    y = data[y_column].to_numpy()

    #  tscv = TimeSeriesSplit()
    #  print(tscv)

    # we want to override n_splits
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=num_splits, test_size=None)
    kf = KFold(n_splits=num_splits, random_state=None, shuffle=False)

    # how are we going to cross-validate them

    # common for all splitting techniques:
    for train_index, test_index in cv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # do split
    #X_train = X[:n_rows]
    #X_test = X[n_rows:]
    #y_train = y[:n_rows]
    #y_test = y[n_rows:]

    # reshape x to ensure it is 2D
    # fixme: already 2D?
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # return all
    return X, y, X_train, y_train, X_test, y_test

class Regressor:
    """Container for analysing different metrics for a single regression class"""
    def __init__(self, cls, fname, training_ratio, **kwargs):
        # construct regressor object
        self.regressor = cls(**kwargs)

        # use load function
        self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test = load(fname, training_ratio)

        # fit data
        self.regressor.fit(self.X_train, self.y_train)

        # get predicted data
        self.y_pred = self.regressor.predict(self.X_test)

    def metric(self, cls, **kwargs) -> float:
        """Takes a sklearn.metrics class and returns the score of the regressor object"""

        # use the metric class to get a score
        return cls(self.y_test, self.y_pred)
