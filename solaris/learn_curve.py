import numpy as np
from time import time

from matplotlib import pyplot as plt

from sklearn.base import clone
from sklearn.utils import check_random_state, shuffle
from sklearn import metrics
from sklearn.externals.joblib import Parallel, delayed


def rmse(y_true, y_pred):
    """Compute root mean squared error from two 1d arrays. """
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


class LearningCurve(object):
    """Creates learning curves by iteratively sampling a subset from
    the given training set, fitting an ``estimator`` on the subset and
    evaluating the estimator on the fixed test set.

    At each iteration the subset size is increased. If
    ``sample_bins`` is an int the subset size is increased by
    ``X_train.shape[0] // sample_bins`` at each iteration until
    ``X_train.shape[0]`` is reached. Otherwise, ``sample_bins`` is expected to
    be an array holding the subset size of each iteration. Each iteration is
    repeated ``num_repetitions`` times using different samples in order
    to create error bars.
    """

    def __init__(self, estimator, sample_bins=10, num_repetitions=10,
                 score_function='mae', random_state=None,
                 n_jobs=1):
        self.base_clf = clone(estimator)
        self.sample_bins = sample_bins
        self.num_repetitions = num_repetitions
        self.score_function = score_function
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs

    def run(self, X_train, y_train, X_test, y_test):
        assert X_train.shape[0] == y_train.shape[0]
        if isinstance(self.sample_bins, int):
            sample_bins = np.linspace(0, X_train.shape[0],
                                      self.sample_bins + 1).astype(np.int)
            sample_bins = sample_bins[1:]
        else:
            sample_bins = np.asarray(self.sample_bins, dtype=np.int)

        train_scores = np.zeros((len(sample_bins), self.num_repetitions))
        test_scores = np.zeros((len(sample_bins), self.num_repetitions))

        self.sample_bins = sample_bins

        out = Parallel(n_jobs=self.n_jobs, verbose=20)(
            delayed(lc_fit)(
                i, j, n_samples, X_train, y_train, X_test, y_test,
                self.base_clf, self.score_function)
                    for i, n_samples in enumerate(sample_bins)
            for j in range(self.num_repetitions))

        # out is a list of 5-tuples (i, n_samples, train_score, test_score)
        for i, j, n_samples, train_score, test_score in out:
            train_scores[i, j] = train_score
            test_scores[i, j] = test_score

        self.train_scores = train_scores
        self.test_scores = test_scores
        print("lc.run() fin")

    def plot(self):
        """Plot the ``train_scores`` and ``test_scores`` using
        error bars.
        """
        fig = plt.figure()
        ax = plt.gca()
        ax.errorbar(self.sample_bins, self.test_scores.mean(axis=1),
                     yerr=self.test_scores.std(axis=1), fmt='r.-',
                     label='test')
        ax.errorbar(self.sample_bins, self.train_scores.mean(axis=1),
                     yerr=self.train_scores.std(axis=1), fmt='b.-',
                     label='train')
        ax.legend(loc='upper right')
        return ax


def lc_fit(i, j, n_samples, X_train, y_train, X_test, y_test, clf,
           score_function):
    if score_function == 'mae':
        score_function_ = metrics.mean_absolute_error
    elif score_function == 'mse':
        score_function_ = metrics.mean_squared_error
    elif score_function == 'rmse':
        score_function_ = rmse
    else:
        score_function_ = rmse

    n = X_train.shape[0]
    idx = shuffle(np.arange(n), random_state=(i * 1000 + j))[:n_samples]
    print 'train %r on %d samples ' % (clf, n_samples)
    t0 = time()
    clf.fit(X_train[idx], y_train[idx])
    print 'took %ds' % (time() - t0)
    train_score = score_function_(y_train[idx], clf.predict(X_train[idx]))
    test_score = score_function_(y_test, clf.predict(X_test))
    print 'test score: %.2f, train_score: %.2f' % (test_score, train_score)
    return (i, j, n_samples, train_score, test_score)


def _transform_data():
    from solaris.run import load_data
    from solaris.models import LocalModel
    from solaris.models import Baseline

    data = load_data()
    X = data['X_train']
    y = data['y_train']

    # no shuffle - past-future split
    offset = X.shape[0] * 0.5
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    print('_' * 80)
    print('transforming data')
    print
    tf = LocalModel(None)
    #tf = Baseline()
    print('transforming train')
    X_train, y_train = tf.transform(X_train, y_train)
    print('transforming test')
    X_test, y_test = tf.transform(X_test, y_test)
    print('fin')

    data = {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test}
    joblib.dump(data, 'data/lcdata.pkl')


if __name__ == '__main__':
    import os
    from sklearn.externals import joblib
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor

    if not os.path.exists('data/lcdata.pkl'):
        _transform_data()
        print('data transformed - re-run script')
    else:
        data = joblib.load('data/lcdata.pkl', mmap_mode='r')
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        print('X_train.shape: (%d, %d)' % X_train.shape)
        print('X_test.shape: (%d, %d)' % X_test.shape)

        est = Ridge(alpha=1.0, normalize=True)
        est = GradientBoostingRegressor(n_estimators=100, verbose=2, max_depth=3,
                                        min_samples_leaf=11, learning_rate=0.2,
                                        random_state=1, loss='ls')

        lc = LearningCurve(est, sample_bins=4, num_repetitions=1,
                           score_function='mae', random_state=1,
                           n_jobs=2)

        print('_' * 80)
        print lc
        print
        print('LC.run')
        lc.run(X_train, y_train, X_test, y_test)
        print('fin')
        print('_' * 80)
        ax = lc.plot()
        import IPython
        IPython.embed()
