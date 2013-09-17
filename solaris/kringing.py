"""
This module implements kringing spatial interpolation.



"""
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcess


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
        x_test = X.station_info[:, [1, 0, 2]]
        n_stations = X.station_info.shape[0]

        X_nm = X.nm
        n_days, n_fx, n_ens, n_hour, n_lat, n_lon = X_nm.shape
        X_nm = X_nm.mean(axis=2)

        out = np.zeros((n_days, n_fx, n_hour, n_stations))
        est = self.est
        for d in range(n_days):
            print 'interpolate day: %d' % d
            for f in range(n_fx):
                for h in range(n_hour):
                    y = X_nm[d, f, h].ravel()
                    est.fit(x, y)
                    interp = est.predict(x_test)
                    out[d, f, h] = interp
        if self.flatten:
            out = out.reshape((out.shape[0], np.prod(out.shape[1:])))

        print 'out.shape', out.shape
        X.blocks['nm_intp'] = out
        return X


class Kringing(Interpolate):

    est = GaussianProcess(corr='squared_exponential',
                          theta0=5.0)


def transform_data():
    from solaris.run import load_data
    from sklearn.externals import joblib

    data = load_data()

    kringing = Kringing()
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
    joblib.dump(data, 'data/interp2_data.pkl')


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
    y = X.nm[day, fx, 0, 3]
    mask = np.ones_like(y, dtype=np.bool)
    rs = np.random.RandomState(1)
    test_idx = np.c_[rs.randint(2, 7, 20),
                     rs.randint(3, 13, 20)]
    print test_idx.shape
    mask[test_idx[:, 0], test_idx[:, 1]] = False
    mask = mask.ravel()
    y = y.ravel()

    #import IPython
    #IPython.embed()

    class KFold(object):

        n_folds = 1

        def __iter__(self):
            yield mask, ~mask

    est = Ridge()
    params = {'normalize': [True, False],
              'alpha': 10.0 ** np.arange(-7, 1, 1)}
    gs = grid_search.GridSearchCV(est, params, cv=KFold(),
                                  loss_func=rmse).fit(x, y)
    print gs.grid_scores_
    print gs.best_score_

    est = GaussianProcess()
    params = {'corr': ['squared_exponential'],
              #'regr': ['constant', 'linear', 'quadratic'],
              'theta0': np.arange(4, 11),
              }

    gs = grid_search.GridSearchCV(est, params, cv=KFold(),
                                  loss_func=rmse).fit(x, y)
    print gs.grid_scores_
    print gs.best_params_
    print gs.best_score_


if __name__ == '__main__':
    transform_data()
    #benchmark()
