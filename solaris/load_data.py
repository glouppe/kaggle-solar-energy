import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import coards

from collections import OrderedDict
from sklearn.externals import joblib

from solaris.sa import StructuredArray


def load_GEFS_data(directory, fx_names, file_sub_str):
    """Loads a list of GEFS files merging them into model format. """
    blocks = OrderedDict()
    for i, f in enumerate(fx_names):
        ds = load_GEFS_file(directory, fx_names[i], file_sub_str)
        X_ds = ds.variables.values()[-1][:]
        if i == 0:
            n_samples = X_ds.shape[0]
            # (n_samples, n_features, n_ensemble, n_hours, lat, long)
            shape = (n_samples, len(fx_names)) + tuple(X_ds.shape[1:])
            print 'create X with shape', shape
            X = np.empty(shape, dtype=np.float32)
        X[:, i, :, :, :] = X_ds
    blocks['nm'] = X
    time = ds.variables['time']
    dates = pd.DatetimeIndex([coards.parse(t, time.units) for t in time])
    blocks['date'] = dates
    X = StructuredArray(blocks)
    X.lat = ds.variables['lat'][:]
    X.lon = ds.variables['lon'][:]
    X.fx_name = OrderedDict()
    X.fx_name['nm'] = fx_names
    X.fx_name['date'] = ['date']
    return X


def load_GEFS_file(directory, data_type, file_sub_str):
    """Loads netCDF file . """
    print 'loading', data_type
    path = os.path.join(directory, data_type + file_sub_str)
    try:
        ds = nc.Dataset(path, 'r+')
    except RuntimeError as e:
        print('cannot load "{0}"'.format(path))
        raise e
    return ds


def load_csv_data(path):
    """Load csv test/train data splitting out times."""
    data = np.loadtxt(path, delimiter=',', dtype=float, skiprows=1)
    times = data[:, 0].astype(int)
    Y = data[:, 1:]
    return times, Y


def main(data_dir='./data', files_to_use='all'):
    if files_to_use == 'all':
        files_to_use = ['dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                        'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc',
                        'pres_msl', 'spfh_2m', 'tcolc_eatm', 'tmax_2m', 'tmin_2m',
                        'tmp_2m', 'tmp_sfc']

    train_sub_str = '_latlon_subset_19940101_20071231.nc'
    test_sub_str = '_latlon_subset_20080101_20121130.nc'

    X_train = load_GEFS_data(os.path.join(data_dir, 'train'),
                             files_to_use, train_sub_str)
    times, y_train = load_csv_data(os.path.join(data_dir, 'train.csv'))

    X_test = load_GEFS_data(os.path.join(data_dir, 'test'),
                            files_to_use, test_sub_str)

    import IPython
    IPython.embed()

    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train}
    joblib.dump(data, 'data/data.pkl')


if __name__ == '__main__':
    main()
