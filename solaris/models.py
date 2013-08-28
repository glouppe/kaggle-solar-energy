import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

from nolearn.dbn import DBN


class FunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ops=(('tcolc_eatm', 'pow', 0.15),)):
        self.fxs = ['dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                    'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc',
                    'pres_msl', 'spfh_2m', 'tcolc_eatm', 'tmax_2m', 'tmin_2m',
                    'tmp_2m', 'tmp_sfc']
        self.ops = ops

    def fit(self, *args, **kw):
        return self

    def transform(self, X, y=None, dates=None):
        for fx_name, op, args in self.ops:
            i = self.fxs.index(fx_name)
            val = X[:, i]
            if op == 'pow':
                val = val ** args
            X[:, i] = val

        out = [X]
        if y is not None:
            out.append(y)
        if dates is not None:
            out.append(dates)

        if len(out) == 1:
            return out[0]
        else:
            return out


class DateExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, op='doy'):
        self.op = op

    def fit(self, X, y=None, dates=None):
        if self.op == 'month' and dates is not None:
            month = dates.map(lambda x: x.month)
            self.lb = LabelBinarizer()
            self.lb.fit(month)
        return self

    def transform(self, X, y=None, dates=None):
        assert dates is not None

        op = self.op
        if op == 'doy':
            vals = dates.map(lambda x: x.dayofyear)
        elif op == 'center':
            doy = dates.map(lambda x: x.dayofyear)
            vals = np.abs(doy - (365.25 / 2.0))
        elif op == 'month':
            month = dates.map(lambda x: x.month)
            vals = self.lb.transform(month)

        vals = vals.astype(np.float32)
        if vals.ndim == 1:
            vals = vals.reshape((vals.shape[0], 1))

        X = np.hstack((X, vals))

        if y is None:
            return X
        else:
            return X, y


class BaselineTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, dates=None):
        # mean over ensembles
        X = X.mean(axis=2)
        # mean over hours
        X = X.mean(axis=2)
        # reshape
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))

        out = [X]
        if y is not None:
            out.append(y)
        if dates is not None:
            out.append(dates)
        if len(out) == 1:
            return out[0]
        else:
            return out


class EnsembleExampleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, dates=None):
        """Creates one row per ensemble member.

        If y is given then copy each value n_ensemble times.
        """
        n_ensemble = X.shape[2]
        # mean over hours
        X = X.mean(axis=3)

        # swap features and ensembles
        X = X.swapaxes(1, 2)
        X = X.reshape(X.shape[0] * X.shape[1], np.prod(X.shape[2:]))

        out = [X]
        if y is not None:
            y = np.repeat(y, n_ensemble, axis=0)
            out.append(y)
        if dates is not None:
            dates = np.repeat(dates, n_ensemble, axis=0)
            out.append(dates)

        if len(out) == 1:
            return out[0]
        else:
            return out


class EnsembledRegressor(BaseEstimator, TransformerMixin):

    def __init__(self, est, date='doy', clip=True):
        self.est = est
        self.trans = EnsembleExampleTransformer()
        self.date = date
        self.clip = clip

    def fit(self, X, y, dates=None, **kw):
        if dates is None:
            X, y = self.trans.transform(X, y)
        else:
            X, y, dates = self.trans.transform(X, y, dates=dates)

        if self.date:
            self.date_extr = DateExtractor(op=self.date)
            self.date_extr.fit(X, dates=dates)
            X = self.date_extr.transform(X, dates=dates)

        print('fit est on X.shape: %s | y.shape: %s' % (
            str(X.shape), str(y.shape)))
        if y.shape[1] == 1:
            y = y.ravel()
        self.est.fit(X, y)
        if self.clip:
            self.clip_high = y.max()
            self.clip_low = y.min()
        return self

    def predict(self, X, dates=None):
        n_ensemble = X.shape[2]
        if dates is None:
            X = self.trans.transform(X)
        else:
            X, dates = self.trans.transform(X, dates=dates)

        if self.date:
            X = self.date_extr.transform(X, dates=dates)

        pred = self.est.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape((pred.shape[0], 1))
        n_stations = pred.shape[1]
        # predict is (n_ensemble * n_days) x n_stations
        pred = pred.reshape((pred.shape[0] / n_ensemble, n_ensemble, n_stations))
        pred = pred.mean(axis=1)
        if self.clip:
            pred = np.clip(pred, self.clip_low, self.clip_high)
        return pred


class Baseline(BaseEstimator, RegressorMixin):

    def __init__(self, est, date='month'):
        self.est = est
        self.date = date

        self.trans = BaselineTransformer()

    def fit(self, X, y, dates=None, **kw):

        self.trans.fit(X, y)
        if dates is None:
            X, y = self.trans.transform(X, y)
        else:
            X, y, dates = self.trans.transform(X, y, dates=dates)

        if self.date:
            self.date_extr = DateExtractor(op=self.date)
            self.date_extr.fit(X, dates=dates)
            X = self.date_extr.transform(X, dates=dates)

        print('fitting est on X.shape: %s' % str(X.shape))
        self.est.fit(X, y)
        return self

    def predict(self, X, dates=None):
        if dates is None:
            X = self.trans.transform(X)
        else:
            X, dates = self.trans.transform(X, dates=dates)
        if self.date:
            X = self.date_extr.transform(X, dates=dates)
        return self.est.predict(X)


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

    def fit(self, X, y, dates=None, X_val=None, y_val=None, yscaler=None):
        self.trans.fit(X)
        X = self.trans.transform(X)
        X = self.scaler.fit_transform(X)

        if X_val is not None:
            X_val = self.trans.transform(X_val)
            X_val = self.scaler.transform(X_val)

        def fine_tune_callback(est, epoch):
            y_pred = est.decision_function(X_val)
            if yscaler is not None:
                y_pred = yscaler.inverse_transform(y_pred)
            try:
                print("Epoch: %d - MAE: %0.2f" %
                  (epoch, metrics.mean_absolute_error(y_val, y_pred)))
            except:
                print('cannot compute val error...')

        est = DBNRegressor(n_hidden_layers=1,
                           n_units=[5000],
                           epochs=200,
                           epochs_pretrain=0,
                           learn_rates_pretrain=[0.0001, 0.001],
                           learn_rates=0.0005,
                           l2_costs_pretrain=0.0,
                           l2_costs=0.0,
                           momentum=0.0,
                           verbose=2,
                           scales=0.01,
                           minibatch_size=64,
                           nest_compare=True,
                           nest_compare_pretrain=True,
                           dropouts=[0.2, 0.5],
                           fine_tune_callback=fine_tune_callback,
                           real_valued_vis=True,
                           fan_outs=[None, None],
                           )
        self.est = est
        print('Train on X.shape: %s' % str(X.shape))
        self.est.fit(X, y)

        return self

    def predict(self, X, dates=None):
        if dates is None:
            X = self.trans.transform(X)
        else:
            X, dates = self.trans.transform(X, dates=dates)
        X = self.scaler.transform(X)
        pred = self.est.predict(X)
        return pred


MODELS = {'baseline': Baseline,
          'ensemble': EnsembledRegressor,
          'dbn': DBNModel}
