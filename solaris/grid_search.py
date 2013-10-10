import numpy as np
import os

from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.preprocessing import StandardScaler


def cluster_encode(X_train, X_test, codebook='kmeans', k=25):
    if codebook == 'kmeans':
        cb = KMeans(k, n_init=1, init='random')
    elif codebook == 'gmm':
        cb = GMM(n_components=k)
    X = np.vstack((X_train, X_test))
    X = StandardScaler().fit_transform(X)
    print('_' * 80)
    print('fitting codebook')
    print
    print cb
    print
    cb.fit(X)
    print 'fin.'
    X_train = cb.transform(X_train)
    X_test = cb.transform(X_test)
    return X_train, X_test


if __name__ == '__main__':


    if not os.path.exists('data/lcdata.pkl'):
        pass
        #_transform_data()
        #print('data transformed - re-run script')
    else:
        data = joblib.load('data/lcdata.pkl', mmap_mode='r')
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        print('X_train.shape: (%d, %d)' % X_train.shape)
        print('X_test.shape: (%d, %d)' % X_test.shape)

        #est = Ridge(alpha=1e-5, normalize=True)
        est = GradientBoostingRegressor(n_estimators=100, verbose=2, max_depth=2,
                                        min_samples_leaf=5, learning_rate=0.1,
                                        random_state=1, loss='ls')

        print('_' * 80)

        idx = np.arange(X_train.shape[0])
        idx = shuffle(idx, random_state=1)[:100000]

        X_train = X_train[idx]
        y_train = y_train[idx]

        X_train, X_test = cluster_encode(X_train, X_test, codebook='kmeans', k=50)
        print X_train.shape

        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        print("MAE:  %0.2f" % metrics.mean_absolute_error(y_test, pred))
        print("RMSE: %0.2f" % np.sqrt(metrics.mean_squared_error(y_test, pred)))
        print("R2: %0.2f" % metrics.r2_score(y_test, pred))
        print('_' * 80)
        import IPython
        IPython.embed()
