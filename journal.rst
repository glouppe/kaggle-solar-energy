=======
Journal
=======


26.8.2013
---------

T1: create one training example per ensemble member;
indicator feature for control and mean.

T2: create multi-task NN architecture; it seems to help to model correlations
in outputs.

F1: differences between points on the lattice (edge features);
    only for important features


Edge Features
 - difference


Node Features
 - NM predictions
 - sum difference
 - max difference
 - daily temp diff
 - daily temp curve

Global Features
 - day of year
 - moon cycle


Q1: summary stats of ensemble or additional training data

Q2: summary stats for time of day (mean, max, std)

Baseline B1:
^^^^^^^^^^^^

Ridge based on mean ensemble daily
10-fold CV: 2260498.69318
PLB:~2200000

5-fold shuffle split: 2262385.80 (+/- 28717.96)


T1: no indicator features
^^^^^^^^^^^^^^^^^^^^^^^^^
5-fold shuffle split: MAE: 2251271.87 (+/- 27283.80)


27.8.2013
---------
DBN on baseline representation (w/o date)

DBN has 98 linear output units.
Scaling Y is important otherwise won't work.

Used the following initial arguments:

n_hidden_layers=3,
n_units=[2000, 2000, 2000],
epochs=200,
epochs_pretrain=10,
learn_rates_pretrain=[0.0001, 0.001, 0.001, 0.001],
learn_rates=0.01,
l2_costs_pretrain=0.000001,
#l2_costs=0.00001,
momentum=0.5,
verbose=2,
scales=0.01,
minibatch_size=200,
#nest_compare=True,
#nest_compare_pretrain=True,
dropouts=[0.2, 0.5, 0.5, 0.5],
fine_tune_callback=fine_tune_callback,
real_valued_vis=True,

These seem to give reasonable results, yet still inferior to Ridge
(2.5M after 100epocs).

First optimize minibatch size then optimize pretraining.

mb32; [2000, 2000, 2000], [0.0001, 0.001, 0.001, 0.001], 0.01

358.606751313
11.5519157874
5.07484053706
46.6158891283

0 4858229.24
10 2843136.65

variables w/ odd distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tcolc_eatm (total condensate): right skewed (bound at 0.0)
tcdc_eatm (total cloud cover, entire): right skewed (bound at 0.0)
apcp_sfc (accumulated precipitation): right skewed (bound at zero)

tmp_2m: slightly left skewed
tmp_sfc: slightly left skewed
ulwrf_sfc: slighly left skewed
tmin_2: slighly left skewed

In [8]: for i, fx in enumerate(files_to_use):

def showit(Z):
    import pylab as pl
    Z = Z.mean(axis=1)
    Z = Z.mean(axis=1)
    pl.figure(); pl.hist(Z.mean(axis=1).mean(axis=1).ravel(), bins=50)


28.8.2013
^^^^^^^^^
Experiments on single Station[0].

GBRT on baseline w/ doy

GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.1, loss='lad',
             max_depth=5, max_features=None, min_samples_leaf=3,
             min_samples_split=2, n_estimators=100, random_state=None,
             subsample=1.0, verbose=3)
obj. function was MAE: 1.4M

Train-test
MAE:  2142931.58
RMSE: 3178710.18

Findings: better than RF; deep trees seems to be needed; date is imp fx
          diff between MAE and RMSE is not very large (so using LS seems ok)


EnsembledRegressor(clip=True, date=center,
          est=Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=True, solver=auto, tol=0.001),
          est__alpha=0.1, est__copy_X=True, est__fit_intercept=True,
          est__max_iter=None, est__normalize=True, est__solver=auto,
          est__tol=0.001)

Train-test
MAE:  2102127.55  <- still better than GBRT
RMSE: 3037915.27


GBRT on baseline w/ doy
GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.1, loss='ls',
             max_depth=5, max_features=None, min_samples_leaf=3,
             min_samples_split=2, n_estimators=200, random_state=None,
             subsample=1.0, verbose=3)
MAE:  2082876.61
RMSE: 3035662.20


Baseline(date=center,
     est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.01, loss=ls,
             max_depth=3, max_features=0.3, min_samples_leaf=3,
             min_samples_split=2, n_estimators=1000, random_state=None,
             subsample=1.0, verbose=1),
     est__alpha=0.9, est__init=None, est__learning_rate=0.01, est__loss=ls,
     est__max_depth=3, est__max_features=0.3, est__min_samples_leaf=3,
     est__min_samples_split=2, est__n_estimators=1000,
     est__random_state=None, est__subsample=1.0, est__verbose=1)

fitting est on X.shape: (3579, 301)
MAE:  1987516.50
RMSE: 3001021.07


Variable transformations
------------------------

it seems that transformations help: eg. diff between two features and ratio


F3: Slope on the grid - both spatial and temporal

F4: feature pooling accross space (time done; doesnt seem to work well)


Baseline(date=center,
     est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.01, loss=ls,
             max_depth=4, max_features=0.3, min_samples_leaf=3,
             min_samples_split=2, n_estimators=1000, random_state=None,
             subsample=1.0, verbose=1),
     est__alpha=0.9, est__init=None, est__learning_rate=0.01, est__loss=ls,
     est__max_depth=4, est__max_features=0.3, est__min_samples_leaf=3,
     est__min_samples_split=2, est__n_estimators=1000,
     est__random_state=None, est__subsample=1.0, est__verbose=1)


fitting est on X.shape: (3579, 2761)
Includes X_l + X_l time mean
MAE:  1962555.41
RMSE: 2965734.97

Baseline(date=center,
     est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.01, loss=ls,
             max_depth=4, max_features=0.3, min_samples_leaf=3,
             min_samples_split=2, n_estimators=1000, random_state=None,
             subsample=1.0, verbose=1),
     est__alpha=0.9, est__init=None, est__learning_rate=0.01, est__loss=ls,
     est__max_depth=4, est__max_features=0.3, est__min_samples_leaf=3,
     est__min_samples_split=2, est__n_estimators=1000,
     est__random_state=None, est__subsample=1.0, est__verbose=1)
fitting est on X.shape: (3579, 2899)
Includes X_l + X_l time mean + kernel_mid
MAE:  1953092.20
RMSE: 2970713.42


________________________________________________________________________________

LocalModel

Train-test

LocalModel(est=RidgeCV(alphas=[  1.00000e-04   1.00000e-03   1.00000e-02   1.00000e-01   1.00000e+00
   1.00000e+01],
    cv=None, fit_intercept=True, gcv_mode=None, loss_func=None,
    normalize=True, score_func=None, scoring=None, store_cv_values=False),
      est__alphas=[  1.00000e-04   1.00000e-03   1.00000e-02   1.00000e-01   1.00000e+00
   1.00000e+01],
      est__cv=None, est__fit_intercept=True, est__gcv_mode=None,
      est__loss_func=None, est__normalize=True, est__score_func=None,
      est__scoring=None, est__store_cv_values=False)


X_nm_l.shape (350742, 15, 5, 5)
(350742, 375) (350742, 98)
(350742, 378) (350742,)
X_nm_l.shape (150332, 15, 5, 5)
(150332, 375) (150332, 98)
# use lat-lon-elev of station
(150332, 378) (150332,)
MAE:  2133943.60
RMSE: 3054626.83
