"""
This module implements kringing spatial interpolation.



"""
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcess


class Interpolate(TransformerMixin, BaseEstimator):

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
                          theta0=10.0)



if __name__ == '__main__':

    from solaris.run import load_data

    data = load_data()
    X = data['X_train']
    y = data['y_train']

    x = Interpolate._grid_data()

    y = X.nm[0, 0, 0, 3].ravel()
    mask = np.ones_like(y, dtype=np.bool)
    test_idx = np.c_[np.random.randint(2, 7, 20),
                     np.random.randint(3, 13, 20)]
    print test_idx.shape
    mask[test_idx[:, 0], test_idx[:, 1]] = False
    mask = mask.ravel()

    #import IPython
    #IPython.embed()


    X_train = x[mask]
    y_train = y[mask]

    X_test = x[~mask]
    y_test = y[~mask]

    class KFold(object):

        n_folds = 1

        def __iter__(self):
            yield mask, ~mask

    est = Ridge()
    params = {'normalize': [True, False],
              'alpha': 10.0 ** np.arange(-7, 1, 1)}
    gs = grid_search.GridSearchCV(est, params, cv=KFold(),
                                  loss_func=metrics.mean_squared_error).fit(x, y)
    print gs.grid_scores_
    print gs.best_score_


    est = GaussianProcess()
    params = {'corr': ['squared_exponential'],
              #'regr': ['constant', 'linear', 'quadratic'],
              'theta0': [10],
              }

    gs = grid_search.GridSearchCV(est, params, cv=KFold(),
                                  loss_func=metrics.mean_squared_error).fit(x, y)
    print gs.grid_scores_
    print gs.best_params_
    print gs.best_score_
