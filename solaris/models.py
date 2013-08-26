import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


class BaselineTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # mean over ensembles
        X = X.mean(axis=2)
        # mean over hours
        X = X.mean(axis=2)
        # reshape
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        return X


class EnsembleExampleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # mean over hours
        X = X.mean(axis=3)

        # swap features and ensembles
        X = X.swapaxes(1, 2)
        X = X.reshape(X.shape[0] * X.shape[1], np.prod(X.shape[2:]))
        return X


class EnsembledRegressor(BaseEstimator, TransformerMixin):

    def __init__(self, est):
        self.est = est
        self.trans = EnsembleExampleTransformer()

    def fit(self, X, y):
        #import IPython
        #IPython.embed()
        n_ensemble = X.shape[2]
        X = self.trans.transform(X)
        y = np.repeat(y, n_ensemble, axis=0)

        self.est.fit(X, y)
        return self

    def predict(self, X):
        n_ensemble = X.shape[2]
        X = self.trans.transform(X)
        pred = self.est.predict(X)
        n_stations = pred.shape[1]
        # predict is (n_ensemble * n_days) x n_stations
        pred = pred.reshape((pred.shape[0] / n_ensemble, n_ensemble, n_stations))
        pred = pred.mean(axis=1)
        return pred


class Baseline(BaseEstimator, RegressorMixin):

    def __init__(self, est):
        self.est = est

        steps = [('trans', BaselineTransformer()), ('est', est)]
        self.pipeline = Pipeline(steps)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        pred = self.pipeline.predict(X)
        return pred


MODELS = {'baseline': Baseline,
          'ensemble': EnsembledRegressor}
