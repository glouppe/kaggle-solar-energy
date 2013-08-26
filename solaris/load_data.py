import csv
import os
import netCDF4 as nc
import numpy as np

from sklearn.externals import joblib


def load_GEFS_data(directory, files_to_use, file_sub_str):
    """Loads a list of GEFS files merging them into model format. """
    for i, f in enumerate(files_to_use):
        block = load_GEFS_file(directory, files_to_use[i], file_sub_str)
        if i == 0:
            n_samples = block.shape[0]
            # (n_samples, n_features, n_ensemble, n_hours, lat, long)
            shape = (n_samples, len(files_to_use)) + tuple(block.shape[1:])
            print 'create X with shape', shape
            X = np.empty(shape, dtype=np.float32)
        X[:, i, :, :, :] = block
    return X


def load_GEFS_file(directory, data_type, file_sub_str):
    """Loads GEFS file using specified merge technique. """
    print 'loading', data_type
    path = os.path.join(directory, data_type + file_sub_str)
    try:
        X = nc.Dataset(path, 'r+').variables.values()[-1][:]
        print path, X.shape
    except RuntimeError as e:
        print('cannot load "{0}"'.format(path))
        raise e
    return X


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
