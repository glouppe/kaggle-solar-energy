import numpy as np
import os

from sklearn.externals import joblib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

from .models import DBNRegressor


def _transform_data():
    from solaris.run import load_data
    from solaris.models import LocalModel

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
    print('transforming train')
    X_train, y_train = tf.transform(X_train, y_train)
    print('transforming test')
    X_test, y_test = tf.transform(X_test, y_test)
    print('fin')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)

    data = {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test}
    joblib.dump(data, 'data/dbndata.pkl')


if __name__ == '__main__':
    if not os.path.exists('data/dbndata.pkl'):
        _transform_data()
        print('data transformed - re-run script')
    else:
        data = joblib.load('data/dbndata.pkl', mmap_mode='r')
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        print('X_train.shape: (%d, %d)' % X_train.shape)
        print('X_test.shape: (%d, %d)' % X_test.shape)

        est = GradientBoostingRegressor(n_estimators=500, verbose=2, max_depth=2,
                                        min_samples_leaf=11, learning_rate=0.1,
                                        max_features=265, subsample=0.5,
                                        random_state=1,
                                        loss='ls')

        print('_' * 80)
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        print("MAE:  %0.6f" % metrics.mean_absolute_error(y_test, pred))
        print("RMSE: %0.6f" % np.sqrt(metrics.mean_squared_error(y_test, pred)))
        print("R2: %0.6f" % metrics.r2_score(y_test, pred))
        print('_' * 80)
        import IPython
        IPython.embed()
