"""
files_to_use = ['dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc',
                'pres_msl', 'spfh_2m', 'tcolc_eatm', 'tmax_2m', 'tmin_2m',
                'tmp_2m', 'tmp_sfc']
fx_names = []
for fx in files_to_use:
    for i in range(9):
        for j in range(16):
            fx_names.append('%s [%d, %d]' % (fx, i, j))
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_fx_imp(est, fx_names, k=50):
    try:
        fx_imp = est.feature_importances_
    except:
        fx_imp = np.abs(est.coef_)

    idx = fx_imp.argsort()[::-1][:k]
    plt.bar(np.arange(idx.shape[0]), fx_imp[idx])
    plt.xticks(np.arange(idx.shape[0]), np.array(fx_names)[idx], rotation=90)

