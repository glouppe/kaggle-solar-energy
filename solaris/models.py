import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from nolearn.dbn import DBN


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
        if y is None:
            return X
        else:
            return X, y


class EnsembleExampleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Creates one row per ensemble member.

        If y is given then copy each value n_ensemble times.
        """
        n_ensemble = X.shape[2]
        # mean over hours
        X = X.mean(axis=3)

        # swap features and ensembles
        X = X.swapaxes(1, 2)
        X = X.reshape(X.shape[0] * X.shape[1], np.prod(X.shape[2:]))

        if y is None:
            return X
        else:
            y = np.repeat(y, n_ensemble, axis=0)
            return X, y


class EnsembledRegressor(BaseEstimator, TransformerMixin):

    def __init__(self, est):
        self.est = est
        self.trans = EnsembleExampleTransformer()

    def fit(self, X, y):
        X, y = self.trans.transform(X, y)
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


class DBNRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_hidden_layers=2, n_units=1000, epochs=100,
                 epochs_pretrain=0, scales=0.05,
                 real_valued_vis=True,
                 use_re_lu=False,
                 uniforms=False,
                 learn_rates_pretrain=0.1,
                 learn_rates=0.1,
                 learn_rate_decays=1.0,
                 learn_rate_minimums=0.0,
                 momentum=0.9,
                 momentum_pretrain=0.9,
                 l2_costs=0.0001,
                 l2_costs_pretrain=0.0001,
                 dropouts=None,
                 minibatch_size=64,
                 verbose=2,
                 fine_tune_callback=None,
                 nest_compare=True,
                 nest_compare_pretrain=None,
                 fan_outs=None,
                 ):
        self.n_hidden_layers = n_hidden_layers
        self.n_units = n_units
        self.epochs = epochs
        self.epochs_pretrain = epochs_pretrain
        self.learn_rates_pretrain = learn_rates_pretrain
        self.learn_rates = learn_rates
        self.learn_rate_decays = learn_rate_decays
        self.l2_costs_pretrain = l2_costs_pretrain
        self.l2_costs = l2_costs
        self.momentum = momentum
        self.momentum_pretrain = momentum_pretrain
        self.verbose = verbose
        self.real_valued_vis = real_valued_vis
        self.use_re_lu = use_re_lu
        self.scales = scales
        self.minibatch_size = minibatch_size
        if dropouts is None:
            dropouts = [0.2] + [0.5] * n_hidden_layers
        self.dropouts = dropouts
        self.fine_tune_callback = fine_tune_callback
        self.nest_compare = nest_compare
        self.nest_compare_pretrain = nest_compare_pretrain
        self.fan_outs = fan_outs

    def fit(self, X, y, X_pretrain=None):
        n_outputs = y.shape[1]

        params = dict(self.__dict__)
        from gdbn.activationFunctions import Linear
        params['output_act_funct'] = Linear()

        n_units = params.pop('n_units')
        n_hidden_layers = params.pop('n_hidden_layers')
        if isinstance(n_units, int):
            units = [n_units] * n_hidden_layers
        else:
            units = n_units
        units = [X.shape[1]] + units + [n_outputs]
        self.dbn = DBN(units, **params)
        self.dbn.fit(X, y, X_pretrain=X_pretrain)

    def predict(self, X):
        return self.dbn.decision_function(X)


class DBNModel(BaseEstimator, RegressorMixin):

    def __init__(self, est=None, trans=None):
        if trans is None:
            trans = BaselineTransformer()
        self.trans = trans
        self.scaler = StandardScaler()

    def fit(self, X, y, X_val=None, y_val=None):
        X = self.trans.transform(X)
        X = self.scaler.fit_transform(X)

        if X_val is not None:
            X_val = self.trans.transform(X_val)
            X_val = self.scaler.transform(X_val)

        def fine_tune_callback(est, epoch):
            y_pred = est.decision_function(X_val)
            try:
                print("Epoch: %d - MAE: %0.2f" %
                  (epoch, metrics.mean_absolute_error(y_val, y_pred)))
            except:
                print('cannot compute val error...')

        est = DBNRegressor(n_hidden_layers=4,
                           n_units=[4000, 4000, 2000, 2000],
                           epochs=200,
                           epochs_pretrain=10,
                           learn_rates_pretrain=[0.0001, 0.001, 0.001, 0.001, 0.001],
                           learn_rates=0.01,
                           l2_costs_pretrain=0.000001,
                           #l2_costs=0.00001,
                           momentum=0.5,
                           verbose=2,
                           scales=0.01,
                           minibatch_size=200,
                           #nest_compare=True,
                           #nest_compare_pretrain=True,
                           dropouts=[0.2, 0.5, 0.5, 0.5, 0.5],
                           fine_tune_callback=fine_tune_callback,
                           real_valued_vis=True,
                           )
        self.est = est
        self.est.fit(X, y)

        return self

    def predict(self, X):
        X = self.trans.transform(X)
        X = self.scaler.transform(X)
        pred = self.est.predict(X)
        return pred


MODELS = {'baseline': Baseline,
          'ensemble': EnsembledRegressor,
          'dbn': DBNModel}
