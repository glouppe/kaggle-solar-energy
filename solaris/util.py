"""
from solaris import util
X_test, y_test = X[offset:], y[offset:]
X_test_ = model.transform(X_test)
y_test_ = model.transform_labels(y_test)
mask = util.clean_missing_labels(y_test)
mask = mask.ravel()
X_test_, y_test_ = X_test_[~mask], y_test_[~mask]
util.plot_deviance(est, X_test_, y_test_)
util.plt.show()
"""


import numpy as np

from sklearn import metrics
from matplotlib import pyplot as plt


def plot_deviance(clf, X_test, y_test):
    """Plot test set deviance of ``clf``.

    Plot train/test deviance at each stage - requires that
    ``clf`` supports ``staged_decision_function``.
    """
    n_estimators = len(clf)
    test_deviance = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        y_test = y_test.ravel()
        y_pred = y_pred.ravel()
        test_deviance[i] = clf.loss_(y_test, y_pred)
        #test_deviance[i] = metrics.mean_absolute_error(y_test, y_pred)

    plt.plot(np.arange(test_deviance.shape[0]) + 1, test_deviance, '-',
             color='r', label='test')
    plt.plot(np.arange(clf.train_score_.shape[0]) + 1, clf.train_score_, '-',
             color='b', label='train')

    plt.xlabel('importance')
    plt.title('Deviance')
    plt.legend(loc='upper left')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Set Deviance')


def clean_missing_labels(y):
    """The function will remove the interpolation done by the comp organizers.

    It removes output values not divisible by 100 because they have been
    interpolated.
    It will also remove forward filled values.
    This function returns a mask that indicates which values in y
    should be retained.
    """
    diffs = np.diff(y, axis=0)
    diffs = np.vstack([np.ones(y.shape[1]), diffs])
    mask = np.logical_or(diffs == 0.0, (y % 100) != 0.0)
    return mask

