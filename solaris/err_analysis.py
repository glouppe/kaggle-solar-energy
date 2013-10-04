import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn import metrics


def err_analysis(y_pred, y_test=None, X_test=None, station_info=None, date=None,
                 mask=None):
    if X_test is not None:
        date = X_test.date
        station_info = pd.DataFrame(data=X_test.station_info,
                                        columns=['nlat', 'elon', 'elev'])
        station_info['stid'] = np.arange(station_info.shape[0])

    if date is None:
        index = pd.date_range('1/1/1994', periods=y_test.shape[0], freq='D')
    else:
        index = date

    ## if station_info is None:
    ##     station_info = pd.read_csv('data/station_info.csv')
    ##     st_cols = station_info['stid']
    ## else:
    ##     st_cols = np.arange(station_info.shape[0])

    if mask is not None:
        y_test[mask] = np.nan

    station_residuals = pd.DataFrame(index=index, columns=station_info['stid'],
                                     data=(y_test - y_pred))

    # # plot MAE by day
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    daily_mae = np.abs(station_residuals).mean(axis=1)
    daily_mae.plot(label='daily MAE', ax=ax1)
    plt.title('Daily MAE')

    # # plot MAE by day of year
    daily_ae = np.abs(station_residuals).sum(axis=1)
    doy = daily_ae.index.map(lambda x: x.dayofyear)
    df = pd.DataFrame(index=daily_ae.index, data={'ae': daily_ae, 'doy': doy})
    doy_mae = df.groupby('doy').mean()

    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
    doy_mae.plot(kind='bar', ax=ax2)
    plt.title('Day-of-year MAE')

    # # plot MAE by month
    daily_ae = np.abs(station_residuals).sum(axis=1)
    month = daily_ae.index.map(lambda x: x.month)
    df = pd.DataFrame(index=daily_ae.index, data={'ae': daily_ae, 'month': month})
    doy_mae = df.groupby('month').mean()
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    doy_mae.plot(kind='bar', ax=ax3)
    plt.title('Monthly MAE')

    # plot MAE by station
    station_mae = np.abs(station_residuals).mean(axis=0)
    station_mae.sort()
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    station_mae.plot(kind='bar', grid=False, ax=ax4)
    plt.title('Station MAE')

    # plot stations on lon-lat grid and colorcode MAE to
    # see spatial-mae correlation
    station_mae.name = 'mae'
    station_info = station_info.join(station_mae, on='stid')
    ax5 = plt.subplot2grid((3, 3), (2, 2))
    cs = ax5.scatter(station_info.elon, station_info.nlat,
                     c=station_info.mae / station_info.mae.max(),
                     s=7**2)
    plt.colorbar(cs, ax=ax5)
    plt.title('Station MAE (spatial correlation)')
    #plt.tight_layout()

    # # plot true-pred correlation
    plt.figure()
    p = y_pred.ravel()
    y = y_test.ravel()
    isfinite = np.isfinite(y)
    p = p[isfinite]
    y = y[isfinite]
    plt.scatter(y, p, s=4)
    plt.plot([0.0, 4e7], [0.0, 4e7], 'k-')
    plt.xlabel('true')
    plt.ylabel('pred')
    plt.title('Scatterplot pred vs true energy output')

    mask = p > y
    upper_mae = metrics.mean_absolute_error(y[mask], p[mask])
    lower_mae = metrics.mean_absolute_error(y[~mask], p[~mask])
    plt.text(0 * 4e7, 0.95 * 4e7, 'Upper MAE: %.0f' % upper_mae, fontsize=8)
    plt.text(0.7 * 4e7, 0 * 4e7, 'Lower MAE: %.0f' % lower_mae, fontsize=8)

    # plot residuals of worst stations
    k = 10
    worst_stid = station_mae.index[-k:]
    f, axes = plt.subplots(k // 2, 2, sharex=True, sharey=True)
    for stid, ax in zip(worst_stid, axes.ravel()):
        station_residuals.iloc[:, stid].plot(ax=ax)
        ax.set_title('Station id %d' % stid)

    plt.show()
