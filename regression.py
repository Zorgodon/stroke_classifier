#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')


def load(fname: str, training_ratio: float) -> tuple:
    # basic spliter for now
    """Loads 2 column data from a CSV file with 2 columns, x and y"""
    # read data
    data = pd.read_csv(fname)

    # get training data
    n_rows = int(len(data) * training_ratio)

    # for one x col only:
    #X = data['x'].to_numpy()

    X_columns = list(data.columns)
    #X_columns.index(y_column) #index
    print(X_columns)
    #.remove
    X_columns.remove(y_column)
    X = data[X_columns].to_numpy()

    y = data['stroke'].to_numpy()

    # do split
    X_train = X[:n_rows]
    X_test = X[n_rows:]
    y_train = y[:n_rows]
    y_test = y[n_rows:]
    
    # reshape x to ensure it is 2D
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    # return all
    return X, y, X_train, y_train, X_test, y_test

X, y, X_train, y_train, X_test, y_test = load('data/400-regression.csv', 0.05)

class Regressor:
    # takes X_train y_train X-test y_test
    def __init__(self, df: pd.DataFrame):
        self.df = df
        print(df)
        print(df['stroke'])

    def linear(self):
        X = df[X_columns].to_numpy()
        y = df[y_column].to_numpy()
       # predict
       # score

    def ridge(self):


dataset=Regressor(df)
