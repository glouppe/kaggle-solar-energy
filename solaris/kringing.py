"""
This module implements kringing spatial interpolation.



"""
import sys
from time import time
import numpy as np
import scipy
import IPython

from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcess


class LinearInterpolator(BaseEstimator):

    def fit(self, x, y):
        self.intp = LinearNDInterpolator(x, y)
        return self

    def predict(self, x):
        return self.intp(x)


class Interpolate(TransformerMixin, BaseEstimator):
    """Interpolate station features.

    This base class interpolates station features by fitting
    an interpolator (ie. estimator) for each 9x16 tile and
    interpolates the 98 station locations.
    The interpolator considers lat, lon, and elev.

    Attributes
    ----------
    est : RegressionMixin
        The estimator that fits a 144 x 3 matrix X and predicts
        a 98 x 3 matrix X_test.
    """

    est = Ridge(normalize=True, alpha=0.1)
    use_nugget = False
    use_mse = False
    grid_mode = False

    def __init__(self, flatten=False):
        self.flatten = flatten

    def fit(self, X, y=None):
        return self

    @classmethod
    def _grid_data(cls):
        from netCDF4 import Dataset
        grid = Dataset('data/gefs_elevations.nc', 'r')
        grid_elev = grid.variables['elevation_control'][:]
        lon = grid.variables['longitude'][:] - 360
        lat = grid.variables['latitude'][:]
        x = np.c_[lon.ravel(), lat.ravel(),
                  grid_elev.ravel()]
        return x

    def transform(self, X, y=None):
        x = self._grid_data()
        lon = np.unique(x[:, 0])
        lat = np.unique(x[:, 1])
        x_test = X.station_info[:, [1, 0, 2]]  # lon, lat, elev
        n_stations = X.station_info.shape[0]

        X_nm = X.nm
        n_days, n_fx, n_ens, n_hour, n_lat, n_lon = X_nm.shape

        pred = np.zeros((n_days, n_fx, n_ens, n_hour, n_stations))
        if self.use_mse:
            sigma2 = np.zeros((n_days, n_fx, n_ens, n_hour, n_stations))
        est = self.est
        for d in range(n_days):
            t0 = time()
            for f in range(n_fx):
                for e in range(n_ens):
                    for h in range(n_hour):
                        y = X_nm[d, f, e, h]
                        if self.grid_mode:
                            est.fit((lon, lat), y)
                        else:
                            est.fit(x, y.ravel())
                        if self.use_mse:
                            c_pred, c_sigma2 = est.predict(x_test, eval_MSE=True)
                            sigma2[d, f, e, h] = c_sigma2
                        else:
                            c_pred = est.predict(x_test)
                        pred[d, f, e, h] = c_pred
            print('interpolate day: %d took %ds' % (d, time() - t0))
        if self.flatten:
            pred = pred.reshape((pred.shape[0], -1))
            if self.use_mse:
                sigma2 = sigma2.reshape((sigma2.shape[0], -1))

        print 'pred.shape', pred.shape
        X.blocks['nm_intp'] = pred
        if self.use_mse:
            X.blocks['nm_intp_sigma'] = np.sqrt(sigma2)
        return X


class Kringing(Interpolate):

    est = GaussianProcess(corr='squared_exponential',
                          theta0=(6.0, 6.0, 6.0))
    use_nugget = False
    use_mse = False


class SplineEstimator(object):

    def fit(self, x, y):
        self.lut = RectBivariateSpline(x[1], x[0], y)
        return self

    def predict(self, X):
        return self.lut.ev(X[:, 1], X[:, 0])


class Spline(Interpolate):

    est = SplineEstimator()
    grid_mode = True


class Linear(Interpolate):

    est = LinearInterpolator()


class MultivariateNormal(object):

    def __init__(self):
        self.lat = scipy.stats.norm(loc=8, scale=2)
        self.lon = scipy.stats.norm(loc=8, scale=2)
        self.elev = scipy.stats.norm(loc=5, scale=2)

    def rvs(self):
        return np.array([self.lon.rvs(), self.lat.rvs(), self.elev.rvs()])


def transform_data():
    from solaris.run import load_data
    from sklearn.externals import joblib

    data = load_data('data/data.pkl')

    #kringing = Kringing()
    #kringing = Spline()
    kringing = Linear()
    print('_' * 80)
    print(kringing)
    print

    for key in ['train', 'test']:
        print('_' * 80)
        print('transforming %s' % key)
        print
        X = data['X_%s' % key]

        X = kringing.fit_transform(X)
        data['X_%s' % key] = X

    print
    print('dumping data')
    joblib.dump(data, 'data/interp7_data.pkl')
    IPython.embed()


def benchmark():
    from solaris.run import load_data
    from sklearn import grid_search
    from sklearn import metrics

    def rmse(y_true, pred):
        return np.sqrt(metrics.mean_squared_error(y_true, pred))

    data = load_data()
    X = data['X_train']
    y = data['y_train']

    x = Interpolate._grid_data()

    fx = 0
    day = 180
    y = X.nm[day, fx].mean(axis=0)[3]
    #nugget = X.nm[day, fx].std(axis=0)[3]
    mask = np.ones_like(y, dtype=np.bool)
    rs = np.random.RandomState(5)
    test_idx = np.c_[rs.randint(2, 7, 20),
                     rs.randint(3, 13, 20)]
    print test_idx.shape
    mask[test_idx[:, 0], test_idx[:, 1]] = False
    mask = mask.ravel()
    y = y.ravel()

    print '_' * 80
    est = GaussianProcess(corr='squared_exponential', theta0=(10, 10, 10))
    est.fit(x[mask], y[mask])
    pred = est.predict(x[~mask])
    print 'MAE: %.2f' % metrics.mean_absolute_error(y[~mask], pred)

    print '_' * 80

    sys.exit(0)

    #import IPython
    #IPython.embed()

    class KFold(object):

        n_folds = 1

        def __iter__(self):
            yield mask, ~mask

        def __len__(self):
            return 1

    est = Ridge()
    params = {'normalize': [True, False],
              'alpha': 10.0 ** np.arange(-7, 1, 1)}
    gs = grid_search.GridSearchCV(est, params, cv=KFold(),
                                  scoring='mean_squared_error').fit(x, y)
    print gs.grid_scores_
    print gs.best_score_

    est = GaussianProcess()
    params = {'corr': ['squared_exponential'],
               'theta0': MultivariateNormal(),
               }

    ## params = {'corr': ['squared_exponential'],
    ##           #'regr': ['constant', 'linear', 'quadratic'],
    ##           'theta0': np.arange(4, 11),
    ##           }

    # gs = grid_search.GridSearchCV(est, params, cv=KFold(),
    #                               loss_func=rmse).fit(x, y)
    gs = grid_search.RandomizedSearchCV(est, params, cv=KFold(),
                                        scoring='mean_squared_error',
                                        n_iter=100).fit(x, y)
    print gs.grid_scores_
    print gs.best_params_
    print gs.best_score_


def inspect():
    from netCDF4 import Dataset
    from matplotlib import pyplot as plt
    from solaris.run import load_data
    data = load_data('data/data.pkl')
    X = data['X_train']
    y = data['y_train']

    ## x_train = Interpolate._grid_data()

    ## fx = 0
    ## day = 180
    ## y_train = X.nm[day, fx, 0, 3]
    ## est = GaussianProcess(corr='squared_exponential',
    ##                       theta0=4.0)
    ## est.fit(x_train, y_train)

    ## n_lat, n_lon = y_train.shape
    ## m = np.mgrid[0:n_lat:0.5, 0:n_lon:0.5]

    grid = Dataset('data/gefs_elevations.nc', 'r')
    lon = np.unique(grid.variables['longitude'][:] - 360)
    lat = np.unique(grid.variables['latitude'][:])

    # take a grid
    for fx_id in range(15):
        G = X.nm[0, fx_id, 0, 3]

        new_lats = np.linspace(lat.min(), lat.max(), 10 * lat.shape[0])
        new_lons = np.linspace(lon.min(), lon.max(), 10 * lon.shape[0])
        new_lats, new_lons = np.meshgrid(new_lats, new_lons)

        x = Interpolate._grid_data()[:, [1, 0]]  # lat, lon
        y = G
        fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(4, 1)
        plt.title('Feature %d' % fx_id)
        ax1.imshow(G, interpolation='none')

        est = GaussianProcess(corr='squared_exponential', theta0=(2.0, 4.0))
        est.fit(x, y.ravel())
        G = est.predict(np.c_[new_lats.ravel(),
                              new_lons.ravel()]).reshape((10 * lon.shape[0],
                                                          10 * lat.shape[0])).T
        ax2.imshow(G, interpolation='none')
        est = SplineEstimator()
        lon = np.unique(x[:, 1])
        lat = np.unique(x[:, 0])
        est.fit((lon, lat), y)
        G = est.predict(np.c_[new_lons.ravel(),
                              new_lats.ravel()]).reshape((10 * lon.shape[0],
                                                          10 * lat.shape[0])).T
        ax3.imshow(G, interpolation='none')

        est = LinearInterpolator()
        est.fit(x, y.ravel())
        G = est.predict(np.c_[new_lats.ravel(),
                              new_lons.ravel()]).reshape((10 * lon.shape[0],
                                                          10 * lat.shape[0])).T
        ax4.imshow(G, interpolation='none')

    IPython.embed()


if __name__ == '__main__':
    transform_data()
    #benchmark()
    #inspect()
