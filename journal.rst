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



LocalModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.1, loss='ls',
             max_depth=4, max_features=0.3, min_samples_leaf=3,
             min_samples_split=2, n_estimators=400, random_state=1,
             subsample=1.0, verbose=1))
Pipeline(steps=[('date', DateTransformer(op='center')), ('ft', FunctionTransformer(block='nm', new_block='nmft',
          ops=(('uswrf_sfc', '/', 'dswrf_sfc'), ('ulwrf_sfc', '/', 'dlwrf_sfc'), ('ulwrf_sfc', '/', 'uswrf_sfc'), ('dlwrf_sfc', '/', 'dswrf_sfc'))))])
FeatureUnion(n_jobs=1,
       transformer_list=[('hm_k2', LocalTransformer(fxs=None, hour_mean=True, k=2)), ('h_k1_3fx', LocalTransformer(fxs={'nm': ['dswrf_sfc', 'uswrf_sfc', 'pwat_eatm']},
         hour_mean=False, k=1))],
       transformer_weights=None)


MAE:  1934130.62
RMSE: 2856188.81
R2: 0.86


7.9.2013
^^^^^^^^

HK_1:




8.9.2013
^^^^^^^^

RidgeCV(alpha=[1e-5, 1e0], normalize=True)

Baseline

CV-5:   2261985.19 (+/- 70066.89)
CV-10:  2255188.94 (+/- 115823.39)
TT 0.7: 2183312.55
TT 0.5: 2303701.03
LB

Local
CV-5
CV-10
TT 0.7: 2084341.24
TT 0.5: 2142705.12
LB

GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.1, loss='ls', max_depth=4, max_features=265,
             min_samples_leaf=3, min_samples_split=2, n_estimators=500,
             random_state=1, subsample=1.0, verbose=2)
Local
CV-5
CV-10
TT 0.7: 1944772.94
TT 0.5: 2027302.29
LB:     2102797.94

seems like gbrt flattens pretty quickly - after 100 trees we already reached
a plateu (2.05).
Since training error is still decreasing I think it might not be a case for
underfitting but rather overfitting - before tuning the learning rate lets look
into min_samples_leaf and max_depth.

12.9.2013
^^^^^^^^^

Ran learning curve experiments. Ridge has lower test error than training,
test error slowly declines with increasing train set size - hardly noticable.
GBRT train error lower than test error (1.73 vs 2.08)


Ideas:

  * Cluster global weather patterns (mixtures of gaussians or k-means)
    - Use as codebook to encode today's weather
  * Use local interpolation
    - use std. linear interpolation
    - look into Kringing literature
  * Feature selection
    - rank each of the 15 features
    - rank positions on local grid
      - create heatmaps from fx importance

13.9.2013
^^^^^^^^^
GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.2, loss='ls', max_depth=4, max_features=None,
             min_samples_leaf=5, min_samples_split=2, n_estimators=100,
             random_state=1, subsample=1.0, verbose=2)
MAE:  2079254.80
RMSE: 3086559.49
R2: 0.84


GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.2, loss='ls', max_depth=5, max_features=None,
             min_samples_leaf=5, min_samples_split=2, n_estimators=100,
             random_state=1, subsample=1.0, verbose=2)
MAE:  2075089.89
RMSE: 3082218.80
R2: 0.84


GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.2, loss='lad', max_depth=4, max_features=None,
             min_samples_leaf=5, min_samples_split=2, n_estimators=100,
             random_state=1, subsample=1.0, verbose=2)

MAE:  2085545.84
RMSE: 3200952.50
R2: 0.83

GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.1, loss='ls', max_depth=2, max_features=None,
             min_samples_leaf=5, min_samples_split=2, n_estimators=100,
             random_state=1, subsample=1.0, verbose=2)

MAE:  2140826.23
RMSE: 3161672.07
R2: 0.84



MAE:  2254349.14
RMSE: 3182497.08
R2: 0.83



initial (k=10, 100 mb size)
MAE:  2689069.89
RMSE: 3742975.79
R2: 0.77

initial (k=20, 1000 mb size)
transform X to shape (2557, 1101)
MAE:  2475579.74
RMSE: 3497225.51
R2: 0.80

three top fx
MAE:  2439879.83
RMSE: 3429030.79
R2: 0.80


transform X to shape (2557, 2461)
MAE:  2242651.84
RMSE: 3169642.89
R2: 0.83


GBRT + encoder + local
Pipeline(steps=[('date', DateTransformer(op='doy')), ('ft', FunctionTransformer(block='nm', new_block='nmft',
          ops=(('ulwrf_sfc', '/', 'dlwrf_sfc'),)))])
EncoderTransformer(codebook='kmeans', ens_mean=True, fx='dswrf_sfc',
          hour_mean=False, k=20, reshape=True)
FeatureUnion(n_jobs=1,
       transformer_list=[('hm_k1', LocalTransformer(aux=True, ens_std=False, fxs=None, hour_mean=True,
         hour_std=False, k=1)), ('h_k2_dswrf_sfc', LocalTransformer(aux=False, ens_std=False, fxs={'nm': ['dswrf_sfc']},
         hour_mean=False, hour_std=False, k=2))],
       transformer_weights=None)
(250586, 375) (250586,)
MAE:  2049997.27
RMSE: 3049374.81
R2: 0.85


GBRT
LocalModel(clip=False,
      est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.1, loss='ls', max_depth=4, max_features=250,
             min_samples_leaf=5, min_samples_split=2, n_estimators=500,
             random_state=1, subsample=1.0, verbose=2))
Pipeline(steps=[('date', DateTransformer(op='doy')), ('ft', FunctionTransformer(block='nm', new_block='nmft',
          ops=(('ulwrf_sfc', '/', 'dlwrf_sfc'),)))])
FeatureUnion(n_jobs=1,
       transformer_list=[('hm_k1', LocalTransformer(aux=True, ens_std=False, fxs=None, hour_mean=True,
         hour_std=False, k=1)), ('h_k2_dswrf_sfc', LocalTransformer(aux=False, ens_std=False, fxs={'nm': ['dswrf_sfc']},
         hour_mean=False, hour_std=False, k=2))],
       transformer_weights=None)
(250586, 275) (250586,)
MAE:  2039388.87
RMSE: 3041438.02
R2: 0.85


GBRT w/ only LocalTransformer(aux=False, ens_std=False, fxs={'nm': ['dswrf_sfc']},
         hour_mean=True, hour_std=False, k=2))]
MAE:  2297962.99
RMSE: 3345037.83
R2: 0.81


GBRT

date + station pos
(250586, 6) (250586,)
MAE:  4542342.97
RMSE: 5659077.89
R2: 0.47

date + station pos + local dswrf_sfc
(250586, 31) (250586,)
MAE:  2257993.58
RMSE: 3303983.09
R2: 0.82

date + station pos + local dswrf_sfc + global dswrf_sfc
(250586, 175) (250586,)
MAE:  2347709.14
RMSE: 3388497.07
R2: 0.81

date + station pos + global dswrf_sfc
X_p.shape:  (250586, 150)
(250586, 150) (250586,)
MAE:  2512085.61
RMSE: 3568142.75
R2: 0.79


Neural Networks
---------------

 * Training data size:

   - global input and single output:

     (n_days * n_ensemble) x (n_fx * n_hours * n_lat * n_lon)

     (2500 * 11) x (15 * 5 * 9 * 16)

      27500 x 10800  <-- too many parameters


-------------

GBRT (scaled y)

MAE:  0.254637
RMSE: 0.380834
R2: 0.845289



15.9.2013
^^^^^^^^^

Kringing experiments

Using Ridge interpolation (alpha=0.1), Ridge fixed
MAE:  2527573.75

Using Kringing interpolation (theta0=4.5, regr=quadratic), Ridge fixed
MAE:  4400000.00


Using Ridge interpolation (alpha=0.1), RidgeCV
MAE:  2462489.94
alpha=0.01

Using Kringing interpolation (theta0=10, regr=const), RidgeCV
MAE:  2227435.74
alpha=1.0

YEAH!!!



Kringing results
----------------
w/o date and station info
w/ hour, mean and sum

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.1, loss='ls', max_depth=4, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=500,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))
MAE:  2051837.90
RMSE: 3045640.26
R2: 0.85

w/ date and station info
w/ hour, mean and sum
KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.1, loss='ls', max_depth=4, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=500,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))
MAE:  2052936.84
RMSE: 3043493.56
R2: 0.85

16.9.2013
---------

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.05, loss='huber', max_depth=5,
             max_features=33, min_samples_leaf=5, min_samples_split=2,
             n_estimators=1000, random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))

[FT] nr new features: 4
(('uswrf_sfc', '/', 'dswrf_sfc'), ('ulwrf_sfc', '/', 'dlwrf_sfc'), ('ulwrf_sfc', '/', 'uswrf_sfc'), ('dlwrf_sfc', '/', 'dswrf_sfc'))
transform to shape:  (250586, 118)
MAE:  2013507.09
RMSE: 3053795.22
R2: 0.85


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.05, loss='ls', max_depth=5, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=1000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))

(('uswrf_sfc', '/', 'dswrf_sfc'), ('ulwrf_sfc', '/', 'dlwrf_sfc'), ('ulwrf_sfc', '/', 'uswrf_sfc'), ('dlwrf_sfc', '/', 'dswrf_sfc'))
transform to shape:  (250586, 118)
MAE:  2042894.75
RMSE: 3036872.99
R2: 0.85


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.05, loss='lad', max_depth=5, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=1000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))
(('uswrf_sfc', '/', 'dswrf_sfc'), ('ulwrf_sfc', '/', 'dlwrf_sfc'), ('ulwrf_sfc', '/', 'uswrf_sfc'), ('dlwrf_sfc', '/', 'dswrf_sfc'))
transform to shape:  (250586, 118)
MAE:  2006458.87
RMSE: 3124858.25
R2: 0.84
-> seems to be the only one where MAE is improving!
MOA trees


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.01, loss='lad', max_depth=5, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=10000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))
(('uswrf_sfc', '/', 'dswrf_sfc'), ('ulwrf_sfc', '/', 'dlwrf_sfc'), ('ulwrf_sfc', '/', 'uswrf_sfc'), ('dlwrf_sfc', '/', 'dswrf_sfc'))
transform to shape:  (250586, 118)
MAE:  1994451.17
RMSE: 3110324.08
R2: 0.84
