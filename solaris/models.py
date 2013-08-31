import numpy as np
import gc

from scipy import signal

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline

from nolearn.dbn import DBN


class ValueTransformer(BaseEstimator, TransformerMixin):
    """Transforms StructuredArray to numpy array. """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.values()
        print('transform X to shape %s' % str(X.shape))
        if y is not None:
            return X, y
        else:
            return X


class FunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, block='nm', ops=(
        ('tcolc_eatm', 'pow', 0.15),
        ('apcp_sfc', 'pow', 0.15),
        ('uswrf_sfc', '/', 'dswrf_sfc'),
        ('ulwrf_sfc', '/', 'dlwrf_sfc'),
        ('ulwrf_sfc', '/', 'uswrf_sfc'),
        ('dlwrf_sfc', '/', 'dswrf_sfc'),
        ('tmax_2m', '-', 'tmin_2m'),
        ('tmp_2m', '-', 'tmp_sfc'),
        ('apcp_sfc', '-', 'pwat_eatm'),
        ('apcp_sfc', '/', 'pwat_eatm'),
        ), new_block='nm_trans'):
        """Apply func transformation to X creating a new block.

        Parameters
        ----------
        block : str
            The name of the block to which ``ops`` are applied.
        new_block : str
            The name of the new block holding the result of ``ops``.
        ops : seq of (left, op, right)
            The sequence of operations to be applied to ``block``.
        """
        self.block = block
        self.ops = ops
        self.new_block = new_block

    def fit(self, *args, **kw):
        return self

    def transform(self, X, y=None, dates=None):
        block = []
        block_name = self.block
        block_fx_names = []
        block_shape = X[self.block].shape
        out = np.empty((block_shape[0], len(self.ops)) +
                        block_shape[2:])
        for i, (left, op, right) in enumerate(self.ops):
            if op == 'pow':
                i = X.fx_name[block_name].index(left)
                vals = X[block_name][:, i]
                vals = vals ** right
            elif op == '/':
                i = X.fx_name[block_name].index(left)
                j = X.fx_name[block_name].index(right)
                vals = (X[block_name][:, i] /
                        X[block_name][:, j])
                # might be infs in there
                vals[~np.isfinite(vals)] = 0.0
            elif op == '-':
                i = X.fx_name[block_name].index(left)
                j = X.fx_name[block_name].index(right)
                vals = (X[block_name][:, i] -
                        X[block_name][:, j])

            out[:, i] = vals
            block_fx_names.append(''.join(map(str, [left, op, right])))

        X[self.new_block] = out
        if hasattr(X, 'fx_names'):
            X.fx_name[self.new_block] = block_fx_names
        if y is not None:
            return X, y
        else:
            return X


class DateTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, op='doy'):
        self.op = op

    def fit(self, X, y=None):
        if self.op == 'month' and hasattr(X, 'date'):
            month = X.date.map(lambda x: x.month)
            self.lb = LabelBinarizer()
            self.lb.fit(month)
        return self

    def transform(self, X, y=None):
        if hasattr(X, 'date'):
            date = X.date
            op = self.op
            if op == 'doy':
                vals = date.map(lambda x: x.dayofyear)
            elif op == 'center':
                doy = date.map(lambda x: x.dayofyear)
                vals = np.abs(doy - (365.25 / 2.0))
            elif op == 'month':
                month = date.map(lambda x: x.month)
                vals = self.lb.transform(month)

            vals = vals.astype(np.float32)
            if vals.ndim == 1:
                vals = vals.reshape((vals.shape[0], 1))

            X['date'] = vals
            if op == 'month':
                X.fx_name['date'] = self.lb.classes_.tolist()
            else:
                X.fx_name['date'] = self.op

        if y is None:
            return X
        else:
            return X, y


class LocalTransformer(BaseEstimator, TransformerMixin):

    k = 2

    def fit(self, X, y=None):
        assert y is not None
        n_stations = y.shape[1]
        self.n_stations = n_stations

        stid = np.arange(98)
        self.stid_lb = LabelBinarizer()
        self.stid_lb.fit(stid)
        return self

    def transform(self, X, y=None, dates=None):
        k = self.k
        n_days = X.shape[0]

        ll_coord = X.station_info[:, 0:2]
        lat_idx = np.searchsorted(X.lat, ll_coord[:, 0])
        lon_idx = np.searchsorted(X.lon, ll_coord[:, 1] + 360)

        blocks = []

        for b_name, X_b in X.blocks.iteritems():
            print 'localizing block: %s' % b_name
            if X_b.ndim == 6:
                X_b = np.mean(X_b, axis=2)
                X_b = np.mean(X_b, axis=2)
                # X_b_l.shape: n_stations * n_days, n_fx, n_lat, n_lon
                X_b_l = np.empty((n_days * self.n_stations, X_b.shape[1],
                                   k * 2 + 1, k * 2 + 1))
                print 'X_b_l.shape', X_b_l.shape

                for i in range(self.n_stations):
                    lai, loi = lat_idx[i], lon_idx[i]
                    local_grid = X_b[:, :, lai - k:lai + k + 1,
                                     loi - k: loi + k + 1]
                    X_b_l[i*n_days:((i+1) * n_days), :, :, :] = local_grid

                X_b_l = X_b_l.reshape((X_b_l.shape[0], np.prod(X_b_l.shape[1:])))
            elif X_b.ndim in (1, 2):
                X_b_l = np.tile(X_b.ravel(), self.n_stations)[:, np.newaxis]
            else:
                raise ValueError('%s has wrong dim: %d' % (b_name, X_b.ndim))

            print b_name, X_b_l.shape
            blocks.append(X_b_l)
            gc.collect()

        del X_b, X_b_l

        ## stid = np.repeat(self.stid_lb.classes_, n_days)
        ## stid_enc = self.stid_lb.transform(stid)
        ## blocks.append(stid_enc)

        # lat, lon, elev
        blocks.append(np.repeat(X.station_info, n_days, axis=0))
        X = np.hstack(blocks)
        del blocks

        if y is not None:
            # flatten y by concatenating cols
            y = y.ravel(1)
            return X, y
        else:
            return X


class LocalGlobalTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, dates=None):
        # mean over ensembles
        X = np.mean(X, axis=2)

        Z = X.copy()

        # mean over hours
        X = X.mean(axis=2)

        Xs = [X, Z]

        kernel_hor = np.array([[-1, 1]])
        X_g = np.empty(X.shape[:2] + (X.shape[2], X.shape[3] - 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_g[i, j, :, :] = signal.convolve2d(X[i, j], kernel_hor,
                                                    mode='valid')
        #Xs.append(X_g)

        kernel_ver = np.array([[-1], [1]])
        X_g = np.empty(X.shape[:2] + (X.shape[2] - 1, X.shape[3]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_g[i, j, :, :] = signal.convolve2d(X[i, j], kernel_ver,
                                                    mode='valid')
        #Xs.append(X_g)

        kernel_small = np.array([[0.25, 0.25], [0.25, 0.25]])
        X_g = np.empty(X.shape[:2] + (X.shape[2] - 1, X.shape[3] - 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_g[i, j, :, :] = signal.convolve2d(X[i, j], kernel_small,
                                                    mode='valid')
        #Xs.append(X_g)

        kernel_mid = np.array([[0.125, 0.125, 0.125],
                               [0.125, 0.0, 0.125],
                               [0.125, 0.125, 0.125]])
        X_g = np.empty(X.shape[:2] + (X.shape[2] - 2, X.shape[3] - 2))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_g[i, j, :, :] = signal.convolve2d(X[i, j], kernel_mid,
                                                    mode='valid')
        Xs.append(X_g)

        Xs = [X_b.reshape(X_b.shape[0], np.prod(X_b.shape[1:])) for X_b in Xs]
        X = np.hstack(Xs)

        out = [X]
        if y is not None:
            out.append(y)
        if dates is not None:
            out.append(dates)
        if len(out) == 1:
            return out[0]
        else:
            return out


class BaselineTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # mean over ensembles
        X_nm = np.mean(X.nm, axis=2)
        # mean over hours
        X_nm = np.mean(X_nm, axis=2)
        # reshape
        X_nm = X_nm.reshape(X_nm.shape[0], np.prod(X_nm.shape[1:]))

        X['nm'] = X_nm
        if y is not None:
            return X, y
        else:
            return X


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

    def mean_predictions(self, pred):
        if pred.ndim == 1:
            pred = pred.reshape((pred.shape[0], 1))
        n_stations = pred.shape[1]
        pred = pred.reshape((pred.shape[0] / 11, 11, n_stations))
        pred = np.mean(pred, axis=1)
        return pred


class EnsembledRegressor(BaseEstimator, TransformerMixin):

    def __init__(self, est, date='center', clip=False):
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
            self.date_extr = DateTransformer(op=self.date)
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
        pred = self.trans.mean_predictions(pred)
        if self.clip:
            pred = np.clip(pred, self.clip_low, self.clip_high)
        return pred


class LocalModel(BaseEstimator, RegressorMixin):

    def __init__(self, est):
        self.est = est
        steps = [('date', DateTransformer(op='center')),
                 ('ft', FunctionTransformer(block='nm', new_block='nmft',
                                            ops=(
                                                ('uswrf_sfc', '/', 'dswrf_sfc'),
                                                ('ulwrf_sfc', '/', 'dlwrf_sfc'),
                                                ('ulwrf_sfc', '/', 'uswrf_sfc'),
                                                ('dlwrf_sfc', '/', 'dswrf_sfc'),
                                                ## ('tmax_2m', '-', 'tmin_2m'),
                                                ## ('tmp_2m', '-', 'tmp_sfc'),
                                                ## ('apcp_sfc', '-', 'pwat_eatm'),
                                                ## ('apcp_sfc', '/', 'pwat_eatm'),
                                                ))),
                 ]
        self.pipeline = Pipeline(steps)
        self.local = LocalTransformer()

    def fit(self, X, y, **kw):
        self.n_stations = y.shape[1]
        X = self.pipeline.transform(X)
        gc.collect()
        self.local.fit(X, y)
        X, y = self.local.transform(X, y)
        gc.collect()
        self.est.fit(X, y)
        return self

    def predict(self, X):
        n_days = X.shape[0]
        X = self.pipeline.transform(X)
        X = self.local.transform(X)
        pred = self.est.predict(X)
        print X.shape, pred.shape
        # reshape - first read days (per station)
        pred = pred.reshape((self.n_stations, n_days)).T
        return pred


class PipelineModel(BaseEstimator, RegressorMixin):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y, **kw):
        if y.shape[1] == 1:
            y = y.ravel()
        self.pipeline.fit(X, y)
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        pred = self.pipeline.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape((pred.shape[0], 1))
        return pred


class Baseline(PipelineModel):

    def __init__(self, est, date='center'):
        self.est = est
        self.date = date

        steps=[('bl_trans', BaselineTransformer())]
        if self.date:
            steps.append(('date', DateTransformer(op=self.date)))

        steps.append(('vals', ValueTransformer()))
        steps.append(('est', est))

        self.pipeline = Pipeline(steps)

        #self.trans = LocalGlobalTransformer()


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
                 nesterov=False,
                 ):
        self.n_hidden_layers = n_hidden_layers
        self.n_units = n_units
        self.epochs = epochs
        self.epochs_pretrain = epochs_pretrain
        self.learn_rates_pretrain = learn_rates_pretrain
        self.learn_rates = learn_rates
        self.learn_rate_decays = learn_rate_decays
        self.learn_rate_minimums = learn_rate_minimums
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
        self.nesterov = nesterov

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
        return self.dbn.chunked_decision_function(X)


class DBNModel(BaseEstimator, RegressorMixin):

    def __init__(self, est=None, transformer='ensemble'):
        if transformer == 'baseline':
            trans = BaselineTransformer()
        elif transformer == 'ensemble':
            trans = EnsembleExampleTransformer()
        self.transformer = transformer
        self.trans = trans
        self.scaler = StandardScaler()

    def fit(self, X, y, dates=None, X_val=None, y_val=None, yscaler=None):
        self.trans.fit(X, y)
        X, y = self.trans.transform(X, y)
        X = self.scaler.fit_transform(X)

        if X_val is not None:
            X_val = self.trans.transform(X_val)
            X_val = self.scaler.transform(X_val)

        def fine_tune_callback(est, epoch):
            y_pred = est.chunked_decision_function(X_val)
            if self.transformer == 'ensemble':
                y_pred = self.trans.mean_predictions(y_pred)
            if yscaler is not None:
                y_pred = yscaler.inverse_transform(y_pred)
            try:
                print("Epoch: %d - MAE: %0.2f" %
                  (epoch, metrics.mean_absolute_error(y_val, y_pred)))
            except:
                print('cannot compute val error...')

        est = DBNRegressor(n_hidden_layers=1,
                           n_units=[1000, 1000],
                           epochs=20,
                           epochs_pretrain=0,
                           learn_rates_pretrain=[0.0001, 0.001, 0.001],
                           learn_rates=0.001, # tuned
                           #learn_rate_decays=0.9,
                           #learn_rate_minimums=0.00001,
                           l2_costs_pretrain=0.0,
                           l2_costs=0.00001,
                           momentum=0.0,
                           verbose=2,
                           scales=0.05, # tuned
                           minibatch_size=64,
                           nesterov=False,
                           nest_compare=True,
                           nest_compare_pretrain=True,
                           dropouts=[0.1, 0.5, 0.5],
                           fine_tune_callback=fine_tune_callback,
                           real_valued_vis=True,
                           fan_outs=[None, 15, None],
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
          'dbn': DBNModel,
          'local': LocalModel}
