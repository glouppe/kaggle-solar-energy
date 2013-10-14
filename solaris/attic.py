import numpy as np
import gc
import IPython

from time import time
from collections import OrderedDict
from scipy import signal
from scipy import ndimage

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GMM
from sklearn.random_projection import GaussianRandomProjection


from .sa import StructuredArray
from . import util


class EncoderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, fx='dswrf_sfc', k=10, reshape=False,
                 codebook='mb_kmeans', ens_mean=False, hour_mean=False):
        self.fx = fx
        self.k = k
        self.reshape = reshape
        self.codebook = codebook
        self.ens_mean = ens_mean
        self.hour_mean = hour_mean

    def fit(self, X, y=None):
        k = self.k
        codebook = self.codebook
        if codebook == 'kmeans':
            cb = KMeans(k, n_init=3, init='k-means++')
        elif codebook == 'gmm':
            cb = GMM(n_components=k, n_iter=200, min_cov=1e-9)
        elif codebook == 'mb_kmeans':
            cb = MiniBatchKMeans(k, batch_size=1000)
        elif codebook == 'random':
            cb = GaussianRandomProjection(n_components=k)
        self.cb_ = cb

        print('_' * 80)
        print('Encoding fx:%s' % self.fx)
        self.fx_idx_ = X.fx_name['nm'].index(self.fx)
        X_nm = X.nm[:, self.fx_idx_].astype(np.float64)

        print('reshaping data')
        if self.hour_mean:
            X_nm = X_nm.mean(axis=2)
            # X = (n_days x n_ens) x (n_lat x n_lon)
            X_nm = X_nm.reshape((np.prod(X_nm.shape[:2]),
                                 np.prod(X_nm.shape[2:])))
        elif self.ens_mean:
            X_nm = X_nm.mean(axis=1)
            # X = (n_days x n_hour) x (n_lat x n_lon)
            X_nm = X_nm.reshape((np.prod(X_nm.shape[:2]),
                                 np.prod(X_nm.shape[2:])))
        else:
            # X = (n_days x n_ens x n_hour) x (n_lat x n_lon)
            X_nm = X_nm.reshape((np.prod(X_nm.shape[:3]),
                                 np.prod(X_nm.shape[3:])))
        print('scaling data')
        self.scaler_ = StandardScaler()
        X_nm = self.scaler_.fit_transform(X_nm)
        print('fitting codebook')
        print
        print cb
        print 'on X.shape: %s' % str(X_nm.shape)
        cb.fit(X_nm)
        print 'fin.'
        return self

    def transform(self, X, y=None):
        X_nm = X.nm[:, self.fx_idx_].astype(np.float64)

        print('reshaping data')
        if self.hour_mean:
            X_nm = X_nm.mean(axis=2)
            # X = (n_days x n_ens) x (n_lat x n_lon)
            X_nm = X_nm.reshape((np.prod(X_nm.shape[:2]),
                                 np.prod(X_nm.shape[2:])))
        elif self.ens_mean:
            X_nm = X_nm.mean(axis=1)
            # X = (n_days x n_hour) x (n_lat x n_lon)
            X_nm = X_nm.reshape((np.prod(X_nm.shape[:2]),
                                 np.prod(X_nm.shape[2:])))
        else:
            # X = (n_days x n_ens x n_hour) x (n_lat x n_lon)
            X_nm = X_nm.reshape((np.prod(X_nm.shape[:3]),
                                 np.prod(X_nm.shape[3:])))

        X_nm = self.scaler_.transform(X_nm)
        try:
            X_nm_tf = self.cb_.transform(X_nm)
        except AttributeError:
            X_nm_tf = self.cb_.predict_proba(X_nm)
        print('transformed X%s to X%s' % (str(X_nm.shape), str(X_nm_tf.shape)))
        if not self.reshape:
            # shape back to n_days x n_ens x n_hour x n_lat x n_lon
            X_nm_tf = X_nm_tf.reshape(X_nm.shape)
        else:
            # shape to n_days x (n_ens x n_hour x n_centroids)
            X_nm_tf = X_nm_tf.reshape((X.nm.shape[0],
                                       np.prod(X_nm_tf.shape) / X.nm.shape[0]))

        print('setting block %s with shape %s' % ('nm_enc_%s' % self.fx,
                                                  X_nm_tf.shape))
        X['nm_enc_%s' % self.fx] = X_nm_tf
        return X


class GlobalTransformer(BaseEstimator, TransformerMixin):
    """Transformer that runs a convolution over the entire grid. """

    def __init__(self, k=5, fxs=None, hour_mean=True):
        self.k = k
        self.hour_mean = hour_mean
        self.fxs = fxs

    def fit(self, X, y=None):
        assert y is not None
        n_stations = y.shape[1]
        self.n_stations = n_stations
        assert X.station_info.shape[0] == self.n_stations

        return self

    def transform(self, X, y=None):
        Xs = []
        for bid, b_name in enumerate(X.blocks):
            X_b = X[b_name]

            # select fx if fxs given
            if self.fxs is not None and b_name in self.fxs:
                fxs = self.fxs[b_name]
                if fxs is not None:
                    idx = [i for i, name in enumerate(X.fx_name[b_name])
                           if name in fxs]
                    X_b = X_b[:, idx]
            else:
                # skip the block
                continue
            print 'GT:', X_b.shape
            # mean over ensembles
            X_b = np.mean(X_b, axis=2)
            # mean over hours
            X_b = np.mean(X_b, axis=2)
            k = self.k
            kernel = np.ones((k, k)) / float(k ** 2.)
            X_g = np.empty(X_b.shape[:2] +
                           ((X_b.shape[2] - (k - 1)) // 2,
                            (X_b.shape[3] - (k - 1)) // 2), dtype=np.float32)
            for i in range(X_b.shape[0]):
                for j in range(X_b.shape[1]):
                    X_g[i, j, :, :] = signal.convolve2d(X_b[i, j], kernel,
                                                        mode='valid')[::2, :-1:2]

            X_g = X_g.reshape(X_g.shape[0], np.prod(X_g.shape[1:]))
            print 'X_g:', X_g.shape
            Xs.append(X_g)
        res = np.hstack(Xs)
        print res.shape
        #res = np.tile(res, (self.n_stations, 1))
        res = np.repeat(res, self.n_stations, axis=0)
        print 'X_g: ', res.shape
        return res


class LocalGlobalTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X_sa, y=None, dates=None):
        print '### in LocalGlobal.transofrm'
        # mean over ensembles
        X = X_sa.nm
        print X.shape
        X = np.mean(X, axis=2)
        #Z = X.copy()

        # mean over hours
        X = X.mean(axis=2)

        Xs = [X]

        kernel_small = np.array([[0.25, 0.25], [0.25, 0.25]])
        X_g = np.empty(X.shape[:2] + (X.shape[2] - 1, X.shape[3] - 1),
                       dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_g[i, j, :, :] = signal.convolve2d(X[i, j], kernel_small,
                                                    mode='valid')
        #Xs.append(X_g)

        kernel_mid = np.array([[0.125, 0.125, 0.125],
                               [0.125, 0.0, 0.125],
                               [0.125, 0.125, 0.125]])
        X_g = np.empty(X.shape[:2] + (X.shape[2] - 2, X.shape[3] - 2),
                       dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_g[i, j, :, :] = signal.convolve2d(X[i, j], kernel_mid,
                                                    mode='valid')
        Xs.append(X_g)

        Xs = [X_b.reshape(X_b.shape[0], np.prod(X_b.shape[1:])) for X_b in Xs]
        X = np.hstack(Xs)
        X_sa.nm = X

        return X_sa


class EnsembleExampleTransformer(BaseEstimator, TransformerMixin):

    hour_mean = False

    def fit(self, X, y=None):
        X = X.nm
        n_ensemble = X.shape[2]
        self.n_ensemble = n_ensemble
        return self

    def transform(self, X, y=None):
        """Creates one row per ensemble member. """
        blocks = OrderedDict()

        for b_name in X.blocks.iterkeys():
            X_b = X[b_name]

            if X_b.ndim >= 3:
                assert X_b.shape[2] == self.n_ensemble

                if self.hour_mean:
                    # mean over hours
                    X_b = X_b.mean(axis=3)

                X_b = X_b.swapaxes(1, 2)
                X_b = X_b.reshape(X_b.shape[0] * X_b.shape[1],
                                  np.prod(X_b.shape[2:]))
            else:
                X_b = np.repeat(X_b, self.n_ensemble, axis=0)

            blocks[b_name] = X_b

        station_info = X.station_info
        X = StructuredArray(blocks)
        X.station_info = station_info
        return X

    def transform_labels(self, y):
        """Copies each value in y n_ensemble times."""
        y = np.repeat(y, self.n_ensemble, axis=0)
        return y

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
        self.ens_tf = EnsembleExampleTransformer()
        self.date_tf = DateTransformer(op='center')
        self.val_tf = ValueTransformer()
        self.clip = clip

    def fit(self, X, y, **kw):
        self.date_tf.fit(X, y)
        self.ens_tf.fit(X, y)
        self.val_tf.fit(X, y)
        print 'after fit'
        X = self.date_tf.transform(X, y)
        X = self.ens_tf.transform(X, y)
        X = self.val_tf.transform(X, y)
        y = self.ens_tf.transform_labels(y)

        print('fit est on X.shape: %s | y.shape: %s' % (
            str(X.shape), str(y.shape)))
        if y.shape[1] == 1:
            y = y.ravel()
        self.est.fit(X, y)
        if self.clip:
            self.clip_high = y.max()
            self.clip_low = y.min()
        return self

    def predict(self, X):
        X = self.date_tf.transform(X)
        X = self.ens_tf.transform(X)
        X = self.val_tf.transform(X)
        pred = self.est.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape((pred.shape[0], 1))
        # predict is (n_ensemble * n_days) x n_stations
        pred = self.ens_tf.mean_predictions(pred)
        if self.clip:
            pred = np.clip(pred, self.clip_low, self.clip_high)
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
        from nolearn.dbn import DBN

        if y.ndim == 2:
            n_outputs = y.shape[1]
        else:
            y = y[:, np.newaxis]
            n_outputs = 1

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
        print X.shape
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
                           scales=0.05,  # tuned
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

