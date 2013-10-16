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

    pertubations = 0
    pertubation_damping_factors = np.array([0.1, 0.1, 10.])

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

        n_stations = X.station_info.shape[0]
        stat_info = X.station_info

        if self.pertubations > 0:
            print('Making %d pertubations to station infos' % self.pertubations)
            rs = np.random.RandomState()
            x_test_control = stat_info
            stat_info = [x_test_control]
            for i in range(self.pertubations):
                x_pertub = (x_test_control + rs.randn(n_stations, 3) *
                            self.pertubation_damping_factors)
                stat_info.append(x_pertub)

            n_stations = n_stations + (self.pertubations * n_stations)
            station_info = np.vstack(stat_info)
            assert station_info.shape[0] == n_stations

        x_test = X.station_info[:, [1, 0, 2]]  # lon, lat, elev

        X_nm = X.nm
        n_days, n_fx, n_ens, n_hour, n_lat, n_lon = X_nm.shape

        X_nm_std = X_nm.std(axis=2)
        X_nm_m = X_nm.mean(axis=2)
        if self.use_nugget:
            from sklearn.gaussian_process.gaussian_process import MACHINE_EPSILON
            nuggets = (X_nm_std / X_nm_m) ** 2.0
            mask = ~np.isfinite(nuggets)
            nuggets[mask] = 10. * MACHINE_EPSILON

        #n_days = 10  # FIXME

        pred = np.zeros((n_days, n_fx, n_hour, n_stations),
                        dtype=np.float32)
        if self.use_mse:
            sigma2 = np.zeros((n_days, n_fx, n_hour, n_stations),
                              dtype=np.float32)
        est = self.est

        print('_' * 80)
        print('computing interpolation')
        print('train x.shape: %s' % str(x.shape))
        print('x_test.shape: %s' % str(x_test.shape))

        for d in range(n_days):
            t0 = time()
            for f in range(n_fx):
                #for e in range(n_ens):
                for h in range(n_hour):
                    #print d, f, h
                    #y = X_nm[d, f, e, h] ## FIXME
                    y = X_nm_m[d, f, h]
                    if self.use_nugget:
                        nugget = nuggets[d, f, h].ravel()
                        est.set_params(nugget=nugget)
                    if self.grid_mode:
                        est.fit((lon, lat), y)
                    else:
                        y = y.ravel()
                        est.fit(x, y)
                    if self.use_mse:
                        c_pred, c_sigma2 = est.predict(x_test, eval_MSE=True)
                        sigma2[d, f, h] = c_sigma2
                    else:
                        c_pred = est.predict(x_test)
                        #pred[d, f, e, h] = c_pred  # FIXME
                    pred[d, f, h] = c_pred
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

    # est = GaussianProcess(corr='squared_exponential',
    #                       theta0=(7.0, 3.0, 3.0))
    est = GaussianProcess(corr='squared_exponential',
                          theta0=(1.0, 0.4, 1.0))  # lon, lat, elev
    use_nugget = True
    use_mse = True


class SplineEstimator(object):

    def fit(self, x, y):
        self.lut = RectBivariateSpline(x[1], x[0], y)
        return self

    def predict(self, X):
        return self.lut.ev(X[:, 1], X[:, 0])


class Spline(Interpolate):

    est = SplineEstimator()
    grid_mode = True


class PertubatedSpline(Spline):

    pertubations = 3


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

    kringing = Kringing()
    #kringing = PertubatedSpline()

    data['description'] = '%r: %r' % (kringing, kringing.est)
    print data['description']

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
    joblib.dump(data, 'data/interp9_data.pkl')
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
    for fx_id in range(3):
        G = X.nm[0, fx_id, 0, 3]

        new_lats = np.linspace(lat.min(), lat.max(), 10 * lat.shape[0])
        new_lons = np.linspace(lon.min(), lon.max(), 10 * lon.shape[0])
        new_lats, new_lons = np.meshgrid(new_lats, new_lons)

        x = Interpolate._grid_data()[:, [1, 0]]  # lat, lon
        y = G
        fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(4, 1)
        plt.title('Feature %d' % fx_id)
        ax1.imshow(G, interpolation='none')

        est = GaussianProcess(corr='squared_exponential', theta0=(3.0, 7.0))
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

    def nugget_kungfu(day=0, fx_id=0, hour=3, theta0=(0.4, 1.0)):
        G = X.nm[day, fx_id, :, hour]

        G_m = G.mean(axis=0)
        G_s = G.std(axis=0)

        from sklearn.gaussian_process.gaussian_process import MACHINE_EPSILON
        nugget = (G_s / G_m) ** 2.0
        mask = ~np.isfinite(nugget)
        nugget[mask] = 10. * MACHINE_EPSILON
        nugget = nugget.ravel()
        est = GaussianProcess(corr='squared_exponential',
                              theta0=theta0,
                              #thetaL=(.5, 1.0), thetaU=(5.0, 10.0),
                              #random_start=100,
                              nugget=nugget,
                              )
        est.fit(x, G_m.ravel())
        print('est.theta_: %s' % str(est.theta_))

        pred, sigma = est.predict(np.c_[new_lats.ravel(), new_lons.ravel()],
                                  eval_MSE=True)
        pred = pred.reshape((10 * lon.shape[0], 10 * lat.shape[0])).T
        sigma = sigma.reshape((10 * lon.shape[0], 10 * lat.shape[0])).T

        fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(4, 1)
        ax1.imshow(G_m, interpolation='none')
        ax1.set_ylabel('Ens mean')
        ax2.imshow(G_s, interpolation='none')
        ax2.set_ylabel('Ens std')
        ax3.imshow(pred, interpolation='none')
        ax3.set_ylabel('GP mean')
        ax4.imshow(sigma, interpolation='none')
        ax4.set_ylabel('GP sigma')


    IPython.embed()


if __name__ == '__main__':
    transform_data()
    #benchmark()
    #inspect()
