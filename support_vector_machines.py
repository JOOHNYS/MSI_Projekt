from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np

class SupportVectorMachines(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, x_train, y_train):

        return self
    
    def predict(self, x_test):

        return [1] * len(x_test)