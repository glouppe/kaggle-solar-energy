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
  --err-analysis   Show error analysis report
"""
import numpy as np
import pandas as pd

from docopt import docopt
from itertools import izip
from time import time

from sklearn.externals import joblib
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn import cross_validation
from sklearn import metrics

from .models import MODELS
from .models import BaselineTransformer
from .models import IndividualEstimator
from .err_analysis import err_analysis
from . import util

# mae_score = metrics.make_scorer(metrics.mean_absolute_error,
#                                 greater_is_better=True)


def load_data(fname='data/interp5_data.pkl'):
    data = joblib.load(fname, mmap_mode='r')
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

    err_analysis(pred, y)
    import IPython
    IPython.embed()


def train_test(args):
    """Run train-test experiment. """
    data = load_data('data/interp6_data.pkl')
    X = data['X_train']
    y = data['y_train']

    # just first 50 stations (otherwise too much)
    ## y = y[:, :25]
    ## X.station_info = X.station_info[:25]

    # no shuffle - past-future split
    offset = X.shape[0] * 0.5
    offset = 3287  # this is 1.1.2003
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    # est = Ridge(alpha=1.0, normalize=True)
    est = GradientBoostingRegressor(n_estimators=2000, verbose=1, max_depth=6,
                                    min_samples_leaf=5, learning_rate=0.02,
                                    max_features=33, random_state=1,
                                    subsample=1.0,
                                    loss='lad')

    model_cls = MODELS[args['<model>']]
    model = model_cls(est=est,
                      with_stationinfo=True,
                      with_date=True, with_solar=False,
                      with_mask=True, with_stationid=False,
                      #intp_blocks=('nm_intp', 'nmft_intp', ),
                      )

    print('_' * 80)
    print('Train-test')
    print
    print model
    print
    print

    scaler = StandardScaler(with_std=False)
    if args['--scaley']:
        y_train = scaler.fit_transform(y_train.copy())

    t0 = time()
    model.fit(X_train, y_train)
    print('model.fit took %.8fm' % ((time() - t0) / 60.))
    pred = model.predict(X_test)
    if args['--scaley']:
        pred = scaler.inverse_transform(pred)

    # FIXME to mask or not to mask
    mask = None
    pred_ = pred.ravel()
    y_test_ = y_test.ravel()
    print('Without masking')
    print("MAE:  %0.2f" % metrics.mean_absolute_error(y_test_, pred_))
    print("RMSE: %0.2f" % np.sqrt(metrics.mean_squared_error(y_test_, pred_)))
    print("R2: %0.2f" % metrics.r2_score(y_test_, pred_))
    print

    mask = util.clean_missing_labels(y_test)
    pred_ = pred.ravel()[~mask.ravel()]
    y_test_ = y_test.ravel()[~mask.ravel()]

    print('_' * 80)
    print('With masking')
    print("MAE:  %0.2f" % metrics.mean_absolute_error(y_test_, pred_))
    print("RMSE: %0.2f" % np.sqrt(metrics.mean_squared_error(y_test_, pred_)))
    print("R2: %0.2f" % metrics.r2_score(y_test_, pred_))

    if args['--err-analysis']:
        # reread test data because has been transformed inplace
        X_test, y_test = X[offset:], y[offset:]
        mask = util.clean_missing_labels(y_test)
        err_analysis(pred, y_test.copy(), X_test=X_test, mask=mask)

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

    est = GradientBoostingRegressor(n_estimators=2000, verbose=1, max_depth=7,
                                    min_samples_leaf=9, learning_rate=0.02,
                                    max_features=20, random_state=1,
                                    subsample=0.5,
                                    loss='lad')

    model_cls = MODELS[args['<model>']]
    model = model_cls(est=est, with_stationinfo=True,
                      with_date=True, with_solar=True,
                      with_mask=True)

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
    print('model.fit took %.fm' % ((time() - t0) / 60.))
    pred = model.predict(X_test)
    if args['--scaley']:
        pred = scaler.inverse_transform(pred)

    data = load_data()
    date_idx = data['X_test'].date
    date_idx = date_idx.map(lambda x: x.strftime('%Y%m%d'))
    stid = pd.read_csv('data/station_info.csv')['stid']
    out = pd.DataFrame(index=date_idx, columns=stid, data=pred)
    out.index.name = 'Date'
    out.to_csv('hk_14.csv')
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
