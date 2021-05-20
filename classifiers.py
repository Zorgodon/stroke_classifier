#!/usr/bin/python
#creating a class that fits a classifier to a dataframe object
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class Classifier:


    def __init__(self, kind: str, dependent: pd):
        #splits the data into training and test sets, turns them into numpy arrays
        self.X_test =

    def logistic(data: pd.DataFrame, ):
        #a method that performs Logistic regression on a data
        regressor = LogisticRegression().fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        score = regressor.score(X_test, y_test)
        return score
