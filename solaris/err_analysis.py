import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def err_analysis(y_true, y_pred, station_info=None, date=None):
    if date is None:
        index = pd.date_range('1/1/1994', periods=y_true.shape[0], freq='D')
    else:
        index = date

    if station_info is None:
        station_info = pd.read_csv('data/station_info.csv')
        st_cols = station_info['stid']
    else:
        st_cols = np.arange(station_info.shape[0])

    station_residuals = pd.DataFrame(index=index, columns=st_cols,
                                     data=(y_true - y_pred))

    # plot MAE by day
    plt.figure()
    daily_mae = np.abs(station_residuals).mean(axis=1)
    daily_mae.plot(label='daily MAE')
    plt.title('Daily MAE')

    # plot MAE by day of year
    daily_ae = np.abs(station_residuals).sum(axis=1)
    doy = daily_ae.index.map(lambda x: x.dayofyear)
    df = pd.DataFrame(index=daily_ae.index, data={'ae': daily_ae, 'doy': doy})
    doy_mae = df.groupby('doy').mean()
    doy_mae.plot(kind='bar', figsize=(24, 8))
    plt.title('Day-of-year MAE')


    # plot MAE by month
    daily_ae = np.abs(station_residuals).sum(axis=1)
    month = daily_ae.index.map(lambda x: x.month)
    df = pd.DataFrame(index=daily_ae.index, data={'ae': daily_ae, 'month': month})
    doy_mae = df.groupby('month').mean()
    plt.figure(figsize=(24, 8))
    doy_mae.plot(kind='bar')
    plt.title('Monthly MAE')

    # plot MAE by station
    plt.figure(figsize=(14, 6))
    station_mae = np.abs(station_residuals).mean(axis=0)
    station_mae.sort()
    station_mae.plot(kind='bar', grid=False)
    plt.title('Station MAE')

    # plot stations on lon-lat grid and colorcode MAE to
    # see spatial-mae correlation
    station_mae.name = 'mae'
    station_info = station_info.join(station_mae, on='stid')
    plt.figure()
    plt.scatter(station_info.elon, station_info.nlat,
                c=station_info.mae / station_info.mae.max())
    plt.colorbar()
    plt.show()
