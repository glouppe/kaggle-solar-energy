import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def err_analysis(y_true, y_pred):
    index = pd.date_range('1/1/1994', periods=5113, freq='D')
    station_info = pd.read_csv('data/station_info.csv')
    station_residuals = pd.DataFrame(index=index, columns=station_info['stid'],
                                     data=(y_true - y_pred))

    # plot MAE by day
    daily_mae = np.abs(station_residuals).mean(axis=1)
    daily_mae.plot(label='daily MAE')

    # plot MAE by day of year
    #daily_mae = np.abs(station_residuals).mean(axis=1)
    #daily_mae.plot(label='daily MAE')

    # plot MAE by month
    #daily_mae = np.abs(station_residuals).mean(axis=1)
    #daily_mae.plot(label='daily MAE')

    # plot MAE by station
    station_mae = np.abs(station_residuals).mean(axis=0)
    station_mae.sort()
    station_mae.plot(kind='bar', grid=False, figsize=(14, 6))

    # plot stations on lon-lat grid and colorcode MAE to
    # see spatial-mae correlation
    station_mae.name = 'mae'
    station_info = station_info.join(station_mae, on='stid')
    plt.figure()
    plt.scatter(station_info.elon, station_info.nlat,
                c=station_info.mae / station_info.mae.max())
    plt.colorbar()
    plt.show()
