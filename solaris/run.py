"""
Solaris

Usage:
  run train_test <model> [options]
  run cross_val <model> [options]
  run grid_search <model> [options]
  run submit <model> [options]
  run inspect
  run -h | --help
  run --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --verbose=LEVEL  Verbosity level [default: 2].
  --n_jobs=N_JOBS  Number of CPUs [default: 1].
  --scaley      Standardize Y before fit.
"""
import numpy as np
import pandas as pd

from docopt import docopt
from itertools import izip
from time import time

from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import metrics
from sklearn import pls

from .models import MODELS
from .models import FunctionTransformer
from .models import DateTransformer
from .models import BaselineTransformer
from .models import PipelineModel
from .models import ValueTransformer
from .models import DBNRegressor
from .models import IndividualEstimator
from .err_analysis import err_analysis


# mae_score = metrics.make_scorer(metrics.mean_absolute_error,
#                                 greater_is_better=True)


def load_data():
    data = joblib.load('data/data.pkl', mmap_mode='r')
    return data


def _cross_val(model, X, y, train, test):
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = metrics.mean_absolute_error(y_test, pred)
    return score, pred


def cross_val(args):
    """Run 5-fold cross-validation. """
    data = load_data()
    X = data['X_train']
    y = data['y_train']

    # just first 50 stations (otherwise too much)
    # y = y[:, :50]
    # X.station_info = X.station_info[:50]

    model_cls = MODELS[args['<model>']]
    # est = RidgeCV(alphas=10.0 ** np.arange(-5, 1, 1), normalize=True)
    est = Ridge(alpha=1e-5, normalize=True)

    model = model_cls(est=est)

    print('_' * 80)
    print('Cross-validation')
    print
    print model
    print
    print
    cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True,
                                random_state=0)

    pool = joblib.Parallel(n_jobs=int(args['--n_jobs']),
                           verbose=int(args['--verbose']))
    res = pool(joblib.delayed(_cross_val)(model, X, y, train, test)
                  for train, test in cv)
    res = list(res)
    scores, preds = zip(*res)
    scores = np.array(scores, dtype=np.float64)
    print("MAE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    pred = np.empty_like(y)
    for (train, test), fold_pred in izip(cv, preds):
        pred[test] = fold_pred

    err_analysis(y, pred)
    import IPython
    IPython.embed()


def train_test(args):
    """Run train-test experiment. """
    data = load_data()
    X = data['X_train']
    y = data['y_train']

    # just first 50 stations (otherwise too much)
    ## y = y[:, :25]
    ## X.station_info = X.station_info[:25]

    # no shuffle - past-future split
    offset = X.shape[0] * 0.5
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    #est = RidgeCV(alphas=10. ** np.arange(-7, 1, 1), normalize=True)
    #est = Ridge(alpha=1e-5, normalize=True)
    # est = RandomForestRegressor(n_estimators=100, verbose=3,
    #                             max_features=0.3, min_samples_leaf=7,
    #                             n_jobs=2, bootstrap=False,
    #                             random_state=1)
    est = GradientBoostingRegressor(n_estimators=1000, verbose=2, max_depth=3,
                                    min_samples_leaf=5, learning_rate=0.1,
                                    max_features=250,
                                    random_state=1,
                                    loss='ls')
    # est = IndividualEstimator(est)
    ## est = Pipeline([('std', StandardScaler()),
    ##                 ('est', KNeighborsRegressor(n_neighbors=5,
    ##                                             weights='distance',
    ##                                             algorithm='auto', ))
    ##                 ])

    model_cls = MODELS[args['<model>']]
    model = model_cls(est=est)

    print('_' * 80)
    print('Train-test')
    print
    print model
    print
    print

    scaler = StandardScaler(with_std=False)
    if args['--scaley']:
        y_train = scaler.fit_transform(y_train.copy())

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if args['--scaley']:
        pred = scaler.inverse_transform(pred)

    print("MAE:  %0.2f" % metrics.mean_absolute_error(y_test, pred))
    print("RMSE: %0.2f" % np.sqrt(metrics.mean_squared_error(y_test, pred)))
    print("R2: %0.2f" % metrics.r2_score(y_test, pred))

    import IPython
    IPython.embed()


def grid_search(args):
    pass


def submit(args):
    """Run train-test experiment. """
    data = load_data()
    X_train = data['X_train']
    y_train = data['y_train']

    X_test = data['X_test']

    ## est = RidgeCV(alphas=10. ** np.arange(-7, -1, 1), normalize=True)
    ## est = Ridge(alpha=1e-5, normalize=True)
    ## est = RandomForestRegressor(n_estimators=25, verbose=3,
    ##                             max_features=0.3, min_samples_leaf=3,
    ##                             n_jobs=1, bootstrap=False)
    est = GradientBoostingRegressor(n_estimators=500, verbose=2, max_depth=4,
                                    min_samples_leaf=3, learning_rate=0.1,
                                    max_features=265,
                                    random_state=1,
                                    loss='ls')


    model_cls = MODELS[args['<model>']]
    model = model_cls(est=est)

    print('_' * 80)
    print('Submit')
    print
    print model
    print
    print

    scaler = StandardScaler()
    if args['--scaley']:
        y_train = scaler.fit_transform(y_train.copy())

    t0 = time()
    model.fit(X_train, y_train)
    print('Model trained in %.fm' % ((time() - t0) / 60.))
    pred = model.predict(X_test)
    if args['--scaley']:
        pred = scaler.inverse_transform(pred)

    data = load_data()
    date_idx = data['X_test'].date
    date_idx = date_idx.map(lambda x: x.strftime('%Y%m%d'))
    stid = pd.read_csv('data/station_info.csv')['stid']
    out = pd.DataFrame(index=date_idx, columns=stid, data=pred)
    out.index.name = 'Date'
    out.to_csv('hk_1.csv')
    import IPython
    IPython.embed()


def inspect(args):
    data = load_data()
    X = data['X_train']
    y = data['y_train']
    import IPython
    IPython.embed()


def main(args):
    if args['train_test']:
        train_test(args)
    elif args['cross_val']:
        cross_val(args)
    elif args['grid_search']:
        grid_search(args)
    elif args['submit']:
        submit(args)
    elif args['inspect']:
        inspect(args)


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    print args
    main(args)
