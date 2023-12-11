import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np

from tqdm import tqdm_notebook

from perceptronito.layer import BaseLayer


class MultilayerPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, n_inputs: int,  layers: list[BaseLayer], epochs: int = 1000):
        self.layers: list[BaseLayer] = layers
        self.epochs = epochs
        for i in range(len(self.layers) - 1):
            self.layers[i].build(n_inputs, next_layer=self.layers[i+1])
            n_inputs = self.layers[i].n_units
        self.layers[-1].build(n_inputs=n_inputs)

    def predict(self, X: np.ndarray):
        pred = X
        for l in self.layers:
            pred = l.process(pred)
        return pred

    def fit(self, X, y):
        for _ in tqdm_notebook(range(self.epochs)):
            for x_i, y_i in zip(X, y):
                self.layers[0].backpropagate(x_i, y_i)
