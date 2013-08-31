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

from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn import metrics
from sklearn import pls
from sklearn.earth import Earth


from .models import MODELS
from .models import FunctionTransformer
from .models import DateTransformer
from .models import BaselineTransformer
from .models import PipelineModel
from .models import ValueTransformer
from .err_analysis import err_analysis


mae_score = metrics.make_scorer(metrics.mean_absolute_error,
                                greater_is_better=True)


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

    model_cls = MODELS[args['<model>']]
    est = Ridge(alpha=0.1, normalize=True)
    #est = pls.CCA(n_components=50, scale=True, max_iter=500, tol=1e-06)
    #est = pls.PLSRegression(n_components=50, scale=True, max_iter=50, tol=1e-06)
    #est = RandomForestRegressor(n_estimators=50, max_features=0.3,
    #                            min_samples_leaf=5, bootstrap=False,
    #                            random_state=13)
    model = model_cls(est=est)

    print('_' * 80)
    print('Cross-validatoin')
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

    # no shuffle - past-future split
    offset = X.shape[0] * 0.7
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    ## est = RidgeCV(alphas=10. ** np.arange(-4, 2, 1), normalize=True)
    ## est = Ridge(alpha=0.1, normalize=True)
    est = RandomForestRegressor(n_estimators=25, verbose=3,
                                max_features=0.3, min_samples_leaf=3,
                                n_jobs=1, bootstrap=False)
    ## est = GradientBoostingRegressor(n_estimators=200, verbose=1, max_depth=4,
    ##                                 min_samples_leaf=3, learning_rate=0.1,
    ##                                 max_features=0.3, random_state=1,
    ##                                 loss='ls')

    model_cls = MODELS[args['<model>']]
    model = model_cls(est=est)

    ## steps = [('baseline', BaselineTransformer()),
    ##          ('date', DateTransformer(op='center')),
    ##          ('ft', FunctionTransformer(block='nm', new_block='nmft')),
    ##          ('val', ValueTransformer()),
    ##          ('est', est)
    ##          ]
    ## model = PipelineModel(Pipeline(steps))

    print('_' * 80)
    print('Train-test')
    print
    print model
    print
    print

    scaler = StandardScaler()
    if args['--scaley']:
        y_train = scaler.fit_transform(y_train)

    model.fit(X_train, y_train,
              X_val=X_test, y_val=y_test, yscaler=scaler)
    pred = model.predict(X_test)
    if args['--scaley']:
        pred = scaler.inverse_transform(pred)

    print("MAE:  %0.2f" % metrics.mean_absolute_error(y_test, pred))
    print("RMSE: %0.2f" % np.sqrt(metrics.mean_squared_error(y_test, pred)))
    import IPython
    IPython.embed()


def grid_search(args):
    pass


def submit(args):
    pass


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
