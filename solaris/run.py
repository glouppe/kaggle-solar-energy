"""
Solaris

Usage:
  run cross_val <model> [options]
  run grid_search <model> [options]
  run submit <model> [options]
  run -h | --help
  run --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --verbose=LEVEL  Verbosity level [default: 2].
  --n_jobs=N_JOBS  Verbosity level [default: 1].
"""
import numpy as np
from docopt import docopt

from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn import cross_validation
from sklearn import metrics


from .models import MODELS


def load_data():
    data = joblib.load('data/data.pkl')
    return data


def cross_val(args):
    data = load_data()
    X = data['X_train']
    y = data['y_train']

    model_cls = MODELS[args['<model>']]
    model = model_cls(est=Ridge(alpha=0.1, normalize=True))

    mae_score = metrics.make_scorer(metrics.mean_absolute_error,
                                    greater_is_better=True)
    print('_' * 80)
    print('Cross-validatoin')
    print
    print model
    print
    print
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=5,
                                       test_size=0.3, random_state=0)
    scores = cross_validation.cross_val_score(model, X, y, cv=cv,
                                              scoring=mae_score,
                                              verbose=int(args['--verbose']),
                                              n_jobs=int(args['--n_jobs']))
    print("MAE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def grid_search(args):
    pass


def submit(args):
    pass


def main(args):
    if args['cross_val']:
        cross_val(args)
    elif args['grid_search']:
        grid_search(args)
    elif args['submit']:
        submit(args)


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    print args
    main(args)
