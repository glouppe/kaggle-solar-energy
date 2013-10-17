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

Added 4 more feature transformations
------------------------------------

('tmax_2m', '-', 'tmin_2m'),
('tmp_2m', '-', 'tmp_sfc'),
('apcp_sfc', '-', 'pwat_eatm'),
('apcp_sfc', '/', 'pwat_eatm'),

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.02, loss='lad', max_depth=6, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=4000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))
MAE:  1980717.16
RMSE: 3083295.06
R2: 0.84


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.05, loss='lad', max_depth=7, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=2000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))

MAE:  1996021.71
RMSE: 3089288.60
R2: 0.84


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.05, loss='lad', max_depth=5, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=1000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))
MAE:  1992978.34
RMSE: 3099598.33
R2: 0.84


12.9.2013
^^^^^^^^^

We are 1st!

Using such a model:


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.02, loss='lad', max_depth=6, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=2000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp'))

it used additional combined features:
('tmax_2m', '/', 'tmin_2m'),
('tmp_2m', '/', 'tmp_sfc'),



nugget and sigma
^^^^^^^^^^^^^^^^

I've compiled a new kriging interpolation dataset (interp3_data).
I've used a nugget and added a sigma block.

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.05, loss='lad', max_depth=5, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=1000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'))
MAE:  1987388.30
RMSE: 3092370.68
R2: 0.84

w/o sigma (interp3_data)
transform to shape:  (250586, 154)
MAE:  1991869.97
RMSE: 3101000.55
R2: 0.84

w/o nugget (interp2_data)
MAE:  1994453.62
RMSE: 3104850.79
R2: 0.84
--> nugget doesnt seem to help (used theta0=4)

lets see how theta0=10 does...

MAE:  1995353.69
RMSE: 3105916.31
R2: 0.84


interp3_data run using sigma w/ hk4 config.

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learn_rate=None,
             learning_rate=0.02, loss='lad', max_depth=6, max_features=33,
             min_samples_leaf=5, min_samples_split=2, n_estimators=2000,
             random_state=1, subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'))
MAE:  1976538.92
RMSE: 3080222.07
R2: 0.84



25.9.2013
^^^^^^^^^

Experimented w/ different max_features values

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=6, max_features=0.3, min_samples_leaf=5,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=1.0, verbose=1),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_global=False, with_stationinfo=True)
MAE:  1983762.13
RMSE: 3095486.37
R2: 0.84


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=6, max_features=0.2, min_samples_leaf=5,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=1.0, verbose=1),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_global=False, with_stationinfo=True)
MAE:  1981673.17  <-- best! (~48 fx)
RMSE: 3095142.36
R2: 0.84


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=6, max_features=0.5, min_samples_leaf=5,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=1.0, verbose=1),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_global=False, with_stationinfo=True)
MAE:  1986963.00
RMSE: 3099572.10
R2: 0.84


0.13.1 vs. gbrt-enh

1981805.33 vs 1981866.25
157.12625185m vs 180.15761642m


28.9.2013
^^^^^^^^^

incorporate solar features:

  - time diff between sun rise and sun set
  - time of sun rise and sun set
  - time of solar zenith
  - solar azimuth at zenith
  - solar declination at zenith
  - solar declination at sun rise
  - solar declination at sun set
  - distance between earth and sun


197 is the mark to beat!


29.9.2013
^^^^^^^^^

HK8

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=7, max_features=20, min_samples_leaf=5,
             min_samples_split=2, n_estimators=3000, random_state=1,
             subsample=1.0, verbose=2),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_solar=False,
       with_stationinfo=True)
3000     1597973.1374


LB MAE 1945755.78


30.8.2013
^^^^^^^^^^

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=35, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=35,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_modmask=False,
       with_solar=False, with_stationid=False, with_stationinfo=False)
MAE:  1971259.05
RMSE: 3074536.46
R2: 0.84


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=35, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=35,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_modmask=True,
       with_solar=True, with_stationinfo=True)
MAE:  1973942.98
RMSE: 3079900.70
R2: 0.84

try solar + modmask but w/o station_info



Mask imputed values
^^^^^^^^^^^^^^^^^^^

The organizeres imputed missing values either by forward filling or
by NN interpolation.
Seems like forward filling created some artefacts that have influence
on the model (see errors of station 78)

I now mask forward filled values and imputed values during model training
and model selection.
It seems that my Held-out errors are now pretty much correlated with the LB error.
In the held-out set there are 1660 masked values.

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=6, max_features=30, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_modmask=True,
       with_solar=True, with_stationid=False, with_stationinfo=True)

MAE:  1933678.49
RMSE: 2977558.93


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=6, max_features=35, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=False)
MAE:  1930890.54
RMSE: 2977140.46
R2: 0.85


w/ 1000 more trees
MAE:  1928320.38
RMSE: 2969441.32
R2: 0.86



KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=8, max_features=20, min_samples_leaf=7,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)
## -- End pasted text --
MAE:  1928355.45
RMSE: 2956186.89
R2: 0.86


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=8, max_features=20, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=8, est__max_features=20,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=False)
MAE:  1927722.99
RMSE: 2958614.68
R2: 0.86


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=35, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=35,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)
MAE:  1930506.85
RMSE: 2975641.68
R2: 0.85


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=35, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=35,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=False)
MAE:  1927897.01
RMSE: 2972463.40
R2: 0.85


1.10.2013
^^^^^^^^^

ran with_stationinfo=False, with_solar=True
    intp_blocks=('nm_intp', 'nmft_intp',)
    min_samples_leaf=11,
    max_depth=9,
    n_estimators=500
1940000.00

-> try with sigmas and


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=9, max_features=20, min_samples_leaf=11,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=9, est__max_features=20,
       est__min_samples_leaf=11, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=True,
       with_stationid=False, with_stationinfo=False)
MAE:  1928053.61
RMSE: 2954749.64
R2: 0.86
-> this completely sucked when submitting a run


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=9, max_features=20, min_samples_leaf=11,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=9, est__max_features=20,
       est__min_samples_leaf=11, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1, intp_blocks=('nm_intp', 'nmft_intp'),
       with_date=True, with_global=False, with_mask=True, with_solar=True,
       with_stationid=False, with_stationinfo=False)
MAE:  1932251.64
RMSE: 2958229.46
R2: 0.86



KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=5, max_features=30, min_samples_leaf=9,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=5, est__max_features=30,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=1000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=True,
       with_stationid=True, with_stationinfo=False)
MAE:  1970857.95
RMSE: 3048997.37
R2: 0.85


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.03, loss=lad,
             max_depth=6, max_features=30, min_samples_leaf=9,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.03,
       est__loss=lad, est__max_depth=6, est__max_features=30,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=1000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=True,
       with_stationid=True, with_stationinfo=False)
MAE:  1950827.00
RMSE: 3003724.22
R2: 0.85


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.04, loss=lad,
             max_depth=6, max_features=30, min_samples_leaf=9,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.04,
       est__loss=lad, est__max_depth=6, est__max_features=30,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=1000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=True,
       with_stationid=True, with_stationinfo=False)
MAE:  1949369.81
RMSE: 2995704.82
R2: 0.85
<- station id gave tiny increase



interp5_data
^^^^^^^^^^^^

mean of ensembles; no nugget and mse
sigma is std of ensembles

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=25, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=25,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)

MAE:  1924173.97
RMSE: 2962615.38
R2: 0.86


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=35, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=35,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=True,
       with_stationid=False, with_stationinfo=True)

MAE:  1925350.19
RMSE: 2963032.26
R2: 0.86

4.10.2013
^^^^^^^^^

Try max_depth=7
Try max_features=20
Try min_leaf=9

Try ens as examples


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=7, max_features=25, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=7, est__max_features=25,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)
MAE:  1920504.15
RMSE: 2950608.79
R2: 0.86


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=7, max_features=20, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=7, est__max_features=20,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.5,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)

MAE:  1918096.81
RMSE: 2947625.04
R2: 0.86




KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=8, max_features=20, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=8, est__max_features=20,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.5,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)
MAE:  1918873.33
RMSE: 2939777.95
R2: 0.86


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=20, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=20,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.5,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)
MAE:  1922856.57
RMSE: 2960597.35
R2: 0.86


9.10.2013
^^^^^^^^^

runs w/o test set masking

interp5
KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
             loss='lad', max_depth=7, max_features=20, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=False)
MAE:  1956862.19
RMSE: 3048263.72
R2: 0.85


interp4
KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=7, max_features=20, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=7, est__max_features=20,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.5,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=False)
MAE:  1966442.06
RMSE: 3062806.62
R2: 0.85


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=7, max_features=20, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=7, est__max_features=20,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.5,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)

MAE:  1958621.37
RMSE: 3047525.77
R2: 0.85
-> again stationinfo doesn't help


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=7, max_features=20, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=7, est__max_features=20,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.5,
       est__verbose=1, intp_blocks=('nm_intp', 'nmft_intp'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)
MAE:  2003582.85
RMSE: 3105179.71
R2: 0.84
-> run using just control member of ensemble


new TT split (2002)

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=20, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.8, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=20,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.8,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)
MAE:  1968803.56
RMSE: 3074546.01
R2: 0.84


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=35, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=35,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)

MAE:  1968489.94
RMSE: 3075962.55
R2: 0.84


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=5, max_features=50, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=5, est__max_features=50,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=False,
       with_solar=False, with_stationid=False, with_stationinfo=True)

MAE:  1974320.20
RMSE: 3088022.21
R2: 0.84

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=100, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=0.5, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=100,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=0.5,
       est__verbose=1, intp_blocks=('nm_intp', 'nmft_intp'),
       with_date=True, with_global=False, with_mask=True, with_solar=False,
       with_stationid=False, with_stationinfo=True)

MAE:  1982593.36
RMSE: 3099118.11
R2: 0.84

w/ masking
MAE:  1936357.36
RMSE: 2981947.05
R2: 0.85


KringingModel(est=RidgeCV(alphas=[  1.00000e-07   1.00000e-06   1.00000e-05   1.00000e-04   1.00000e-03
   1.00000e-02   1.00000e-01   1.00000e+00],
    cv=None, fit_intercept=True, gcv_mode=None, loss_func=None,
    normalize=True, score_func=None, scoring=None, store_cv_values=False),
       est__alphas=[  1.00000e-07   1.00000e-06   1.00000e-05   1.00000e-04   1.00000e-03
   1.00000e-02   1.00000e-01   1.00000e+00],
       est__cv=None, est__fit_intercept=True, est__gcv_mode=None,
       est__loss_func=None, est__normalize=True, est__score_func=None,
       est__scoring=None, est__store_cv_values=False,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=False,
       with_solar=False, with_stationid=False, with_stationinfo=True)

With masking
MAE:  2129490.69
RMSE: 3020393.59
R2: 0.85

Without masking
MAE:  2172092.17
RMSE: 3125587.30
R2: 0.84

Baseline(date=center,
     est=RidgeCV(alphas=[  1.00000e-07   1.00000e-06   1.00000e-05   1.00000e-04   1.00000e-03
   1.00000e-02   1.00000e-01   1.00000e+00],
    cv=None, fit_intercept=True, gcv_mode=None, loss_func=None,
    normalize=True, score_func=None, scoring=None, store_cv_values=False),
     est__alphas=[  1.00000e-07   1.00000e-06   1.00000e-05   1.00000e-04   1.00000e-03
   1.00000e-02   1.00000e-01   1.00000e+00],
     est__cv=None, est__fit_intercept=True, est__gcv_mode=None,
     est__loss_func=None, est__normalize=True, est__score_func=None,
     est__scoring=None, est__store_cv_values=False)

With masking
MAE:  2227701.24
RMSE: 3109077.66
R2: 0.84

Without masking
MAE:  2269232.80
RMSE: 3207800.21
R2: 0.83


KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=33,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=False,
       with_solar=False, with_stationid=False, with_stationinfo=True)
With masking
MAE:  1922762.48
RMSE: 2956874.92
R2: 0.85

Without masking
MAE:  1968845.32
RMSE: 3074189.50
R2: 0.84


11.10.2013
^^^^^^^^^^

Used new interpolation technique (b-spline)

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=33,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=False,
       with_solar=False, with_stationid=False, with_stationinfo=True)
Without masking
MAE:  1961824.81
RMSE: 3068296.24
R2: 0.84

________________________________________________________________________________
With masking
MAE:  1915553.96
RMSE: 2950142.98
R2: 0.85


Ideas (Gilles):

  1) Add new samples for "fake stations" around the actual target stations and
  set the same output. => This should make the model more robust by forcing the
  solar activity to be locally the same

  2) Make multiple predictions around the target stations and average. => This
  should reduce the variance of predictions

  Combine 1 + 2.

  3) Study the variance of the GBRT model
      - try several random seeds
      - probe the neighborhood of the stations

  4) Filter out unimportant features

  5) Features
      - add "wrapped" doy:
          (doy + offset) % 365 for offset = 90, 180, 270?
      - is there a feature for the hour of the measurement?



12.10.2013
^^^^^^^^^^

Q: Is there variance in GBRT models? Yes! The following ones uses
random_state=2 instead of 1 and improves accuracy.

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=2,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=33,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=2, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_global=False, with_mask=False,
       with_solar=False, with_stationid=False, with_stationinfo=True)

Without masking
MAE:  1959680.30
RMSE: 3064633.13
R2: 0.84

With masking
MAE:  1913301.65
RMSE: 2946062.64
R2: 0.85


PertubedKrigingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=10, min_samples_leaf=9,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=0.5, verbose=1),
           est__alpha=0.9, est__init=None, est__learning_rate=0.02,
           est__loss=lad, est__max_depth=6, est__max_features=10,
           est__min_samples_leaf=9, est__min_samples_split=2,
           est__n_estimators=1000, est__random_state=1, est__subsample=0.5,
           est__verbose=1, intp_blocks=('nm_intp', 'nmft_intp'),
           with_date=True, with_mask=False, with_solar=False,
           with_stationinfo=None)


Without masking
MAE:  1981755.23
RMSE: 3104136.39
R2: 0.84

With masking
MAE:  1935107.09
RMSE: 2986469.65
R2: 0.85



EnsembleKrigingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.04, loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=9,
             min_samples_split=2, n_estimators=500, random_state=1,
             subsample=0.2, verbose=1),
           est__alpha=0.9, est__init=None, est__learning_rate=0.04,
           est__loss=lad, est__max_depth=6, est__max_features=33,
           est__min_samples_leaf=9, est__min_samples_split=2,
           est__n_estimators=500, est__random_state=1, est__subsample=0.2,
           est__verbose=1, intp_blocks=('nm_intp', 'nmft_intp'),
           with_date=True, with_mask=False, with_solar=False,
           with_stationinfo=True)

n_nes = 11

transform to shape: (3543386, 154)
dtype: float32
size of X in mb: 2081.61
________________________________________________________________________________
fitting estimator...

      Iter       Train Loss      OOB Improve   Remaining Time
         1     6676791.1588      215820.6106         1938.11m
         2     6470898.3519      209119.1461         1386.85m
         3     6278283.1402      199989.8206         1202.14m
         4     6078670.9198      192223.0153         1109.00m
         5     5897917.1003      181538.4845         1050.97m
         6     5720301.9820      175039.3000         1011.32m
         7     5557389.6499      164459.1493          982.20m
         8     5388759.2654      164213.0084          962.65m
         9     5241261.0758      153626.6735          946.51m
        10     5090505.7371      144907.6095          930.70m
        20     3924466.8644       97192.9144          844.55m
        30     3208946.8315       54664.3455          772.27m
        40     2769488.7391       34763.0937          729.23m
        50     2501605.6560       19688.3076          697.99m
        60     2353711.1177       12184.4013          672.18m
        70     2252526.4211        7368.7132          649.64m
        80     2201595.0008        4771.9175          629.23m
        90     2166232.2779        2679.4130          610.27m
       100     2139461.2452        2205.7733          592.13m
       200     2031123.5678         498.0011          431.68m
       300     1983349.3872         259.1840          281.38m
       400     1949273.8976         344.0977          138.05m
       500     1920733.0446         254.6645            0.00s
model.fit took 708.77068682m
[FT] nr new features: 10
transform to shape: (1968428, 154)
dtype: float32
size of X in mb: 1156.38
(1968428,)
Without masking
MAE:  1967450.65
RMSE: 3071018.65
R2: 0.84

________________________________________________________________________________
With masking
MAE:  1921013.72
RMSE: 2952923.20
R2: 0.85
=======


15.10.2013
^^^^^^^^^^

Variable importances with ExtraTreesClassifier(max_features=1, n_estimators=500)

82 nmft_intp_uswrf_sfc/dswrf_sfc_3 0.0201937944785
221 nmft_intp_ulwrf_sfc/uswrf_sfc_m 0.0184327024786
19 nm_intp_ulwrf_sfc_0 0.0166179925147
204 nm_intp_dswrf_sfc_m 0.0147316672135
208 nm_intp_ulwrf_tatm_m 0.0134287797581
222 nmft_intp_dlwrf_sfc/dswrf_sfc_m 0.0133668550344
49 nm_intp_spfh_2m_0 0.0132272063518
81 nmft_intp_uswrf_sfc/dswrf_sfc_2 0.0121702599986
2 lon 0.0114467669466
34 nm_intp_tcdc_eatm_0 0.0111164968979
43 nm_intp_apcp_sfc_4 0.0108485662755
0 doy 0.0105541536685
47 nm_intp_pres_msl_3 0.0102945869151
61 nm_intp_tmax_2m_2 0.0101036267138
21 nm_intp_ulwrf_sfc_2 0.00998009184493
31 nm_intp_pwat_eatm_2 0.00990288483034
36 nm_intp_tcdc_eatm_2 0.00968353332305
216 nm_intp_tmin_2m_m 0.0095803932702
37 nm_intp_tcdc_eatm_3 0.00941401451949
97 nmft_intp_dlwrf_sfc/dswrf_sfc_3 0.00895560624953
218 nm_intp_tmp_sfc_m 0.0088455519542
13 nm_intp_dlwrf_sfc_4 0.00883892848454
64 nm_intp_tmin_2m_0 0.00876455711075
33 nm_intp_pwat_eatm_4 0.00871410789317
67 nm_intp_tmin_2m_3 0.00849656247567
17 nm_intp_uswrf_sfc_3 0.00849278397754
77 nm_intp_tmp_sfc_3 0.00844515501974
60 nm_intp_tmax_2m_1 0.00788425898625
39 nm_intp_apcp_sfc_0 0.00781172926849
32 nm_intp_pwat_eatm_3 0.00777968299496
22 nm_intp_ulwrf_sfc_3 0.00768889770269
30 nm_intp_pwat_eatm_1 0.00763521845828
76 nm_intp_tmp_sfc_2 0.00763491150461
73 nm_intp_tmp_2m_4 0.00761612370818
63 nm_intp_tmax_2m_4 0.007450430372
15 nm_intp_uswrf_sfc_1 0.00719447178178
5 nm_intp_dswrf_sfc_1 0.0071074174592
68 nm_intp_tmin_2m_4 0.00702503511129
7 nm_intp_dswrf_sfc_3 0.00676053439573
78 nm_intp_tmp_sfc_4 0.00662875753248
62 nm_intp_tmax_2m_3 0.0065817349919
207 nm_intp_ulwrf_sfc_m 0.00656091963796
209 nm_intp_pwat_eatm_m 0.00654646627066
220 nmft_intp_ulwrf_sfc/dlwrf_sfc_m 0.00653745102788
213 nm_intp_spfh_2m_m 0.00635121388898
94 nmft_intp_dlwrf_sfc/dswrf_sfc_0 0.00622656493176
35 nm_intp_tcdc_eatm_1 0.00621402454024
18 nm_intp_uswrf_sfc_4 0.00614279893571
66 nm_intp_tmin_2m_2 0.00601591372873
24 nm_intp_ulwrf_tatm_0 0.00599549721072
1 lat 0.00597638159923
69 nm_intp_tmp_2m_0 0.00589018415456
28 nm_intp_ulwrf_tatm_4 0.00570474681343
65 nm_intp_tmin_2m_1 0.00568109565041
54 nm_intp_tcolc_eatm_0 0.00564527044306
58 nm_intp_tcolc_eatm_4 0.00554195519771
231 nm_intp_sigma_uswrf_sfc_sigma_m 0.00551955457996
9 nm_intp_dlwrf_sfc_0 0.00545848139352
217 nm_intp_tmp_2m_m 0.00541578923872
145 nm_intp_sigma_ulwrf_sfc_sigma_1 0.00539091683823
212 nm_intp_pres_msl_m 0.00533917230882
75 nm_intp_tmp_sfc_1 0.00528636655266
205 nm_intp_dlwrf_sfc_m 0.00524991927462
38 nm_intp_tcdc_eatm_4 0.00524608978298
189 nm_intp_sigma_tmin_2m_sigma_0 0.00522054437044
127 nmft_intp_apcp_sfc/pwat_eatm_3 0.00516102667974
8 nm_intp_dswrf_sfc_4 0.00507387908154
46 nm_intp_pres_msl_2 0.00504016342089
104 nmft_intp_tmax_2m/tmin_2m_0 0.00503170762072
227 nmft_intp_apcp_sfc-pwat_eatm_m 0.00500665399438
48 nm_intp_pres_msl_4 0.00499386717095
215 nm_intp_tmax_2m_m 0.00491950112478
120 nmft_intp_apcp_sfc-pwat_eatm_1 0.00486916852115
12 nm_intp_dlwrf_sfc_3 0.0047527211005
190 nm_intp_sigma_tmin_2m_sigma_1 0.00473492790777
100 nmft_intp_tmax_2m-tmin_2m_1 0.00467949965669
233 nm_intp_sigma_ulwrf_tatm_sigma_m 0.00466706774344
93 nmft_intp_ulwrf_sfc/uswrf_sfc_4 0.00461991576951
224 nmft_intp_tmax_2m/tmin_2m_m 0.00453547142302
90 nmft_intp_ulwrf_sfc/uswrf_sfc_1 0.00452090631869
117 nmft_intp_tmp_2m/tmp_sfc_3 0.00443198217978
45 nm_intp_pres_msl_1 0.0043353596124
183 nm_intp_sigma_tcolc_eatm_sigma_4 0.0043188349121
110 nmft_intp_tmp_2m-tmp_sfc_1 0.00431590812511
3 elev 0.00421414148382
133 nm_intp_sigma_dswrf_sfc_sigma_4 0.00420421544519
125 nmft_intp_apcp_sfc/pwat_eatm_1 0.00417051570219
152 nm_intp_sigma_ulwrf_tatm_sigma_3 0.0041519561992
130 nm_intp_sigma_dswrf_sfc_sigma_1 0.00413799254827
50 nm_intp_spfh_2m_1 0.00412872451117
116 nmft_intp_tmp_2m/tmp_sfc_2 0.00409090669391
239 nm_intp_sigma_tcolc_eatm_sigma_m 0.00408980305341
51 nm_intp_spfh_2m_2 0.00408767695983
235 nm_intp_sigma_tcdc_eatm_sigma_m 0.00406255444551
188 nm_intp_sigma_tmax_2m_sigma_4 0.00399381867402
223 nmft_intp_tmax_2m-tmin_2m_m 0.00391462212918
25 nm_intp_ulwrf_tatm_1 0.00385027303896
23 nm_intp_ulwrf_sfc_4 0.00384998444919
53 nm_intp_spfh_2m_4 0.00383535725915
173 nm_intp_sigma_pres_msl_sigma_4 0.00382548082641
16 nm_intp_uswrf_sfc_2 0.00381969103952
210 nm_intp_tcdc_eatm_m 0.00380444791989
107 nmft_intp_tmax_2m/tmin_2m_3 0.0037831790158
52 nm_intp_spfh_2m_3 0.00377840261987
84 nmft_intp_ulwrf_sfc/dlwrf_sfc_0 0.00376236675446
27 nm_intp_ulwrf_tatm_3 0.00374116081961
187 nm_intp_sigma_tmax_2m_sigma_3 0.003679297229
198 nm_intp_sigma_tmp_2m_sigma_4 0.00367862710239
177 nm_intp_sigma_spfh_2m_sigma_3 0.00366481745879
103 nmft_intp_tmax_2m-tmin_2m_4 0.00364137346522
57 nm_intp_tcolc_eatm_3 0.00362550070667
126 nmft_intp_apcp_sfc/pwat_eatm_2 0.00357619551676
80 nmft_intp_uswrf_sfc/dswrf_sfc_1 0.00351669586597
225 nmft_intp_tmp_2m-tmp_sfc_m 0.00347301656102
113 nmft_intp_tmp_2m-tmp_sfc_4 0.00340840799503
148 nm_intp_sigma_ulwrf_sfc_sigma_4 0.00340093378075
182 nm_intp_sigma_tcolc_eatm_sigma_3 0.00338332015748
171 nm_intp_sigma_pres_msl_sigma_2 0.00332214807022
134 nm_intp_sigma_dlwrf_sfc_sigma_0 0.00329916993907
159 nm_intp_sigma_tcdc_eatm_sigma_0 0.00329305322133
172 nm_intp_sigma_pres_msl_sigma_3 0.00329166452225
230 nm_intp_sigma_dlwrf_sfc_sigma_m 0.00326916628537
161 nm_intp_sigma_tcdc_eatm_sigma_2 0.00326487292039
243 nm_intp_sigma_tmp_sfc_sigma_m 0.00325128361056
240 nm_intp_sigma_tmax_2m_sigma_m 0.00319437084984
163 nm_intp_sigma_tcdc_eatm_sigma_4 0.00317942490258
85 nmft_intp_ulwrf_sfc/dlwrf_sfc_1 0.00316679353366
20 nm_intp_ulwrf_sfc_1 0.00315540841872
123 nmft_intp_apcp_sfc-pwat_eatm_4 0.00313330987893
42 nm_intp_apcp_sfc_3 0.00309260664906
174 nm_intp_sigma_spfh_2m_sigma_0 0.00308580644865
164 nm_intp_sigma_apcp_sfc_sigma_0 0.00307980451156
238 nm_intp_sigma_spfh_2m_sigma_m 0.00307841982906
162 nm_intp_sigma_tcdc_eatm_sigma_3 0.0030499631853
176 nm_intp_sigma_spfh_2m_sigma_2 0.00302420531394
40 nm_intp_apcp_sfc_1 0.00301481937758
72 nm_intp_tmp_2m_3 0.00295201034695
241 nm_intp_sigma_tmin_2m_sigma_m 0.00289838602574
201 nm_intp_sigma_tmp_sfc_sigma_2 0.00288738353434
200 nm_intp_sigma_tmp_sfc_sigma_1 0.00287371099469
96 nmft_intp_dlwrf_sfc/dswrf_sfc_2 0.00284389492564
170 nm_intp_sigma_pres_msl_sigma_1 0.00283943458197
234 nm_intp_sigma_pwat_eatm_sigma_m 0.00281315841313
226 nmft_intp_tmp_2m/tmp_sfc_m 0.00279594918922
156 nm_intp_sigma_pwat_eatm_sigma_2 0.00277420142522
237 nm_intp_sigma_pres_msl_sigma_m 0.00277119807865
70 nm_intp_tmp_2m_1 0.00275157698385
180 nm_intp_sigma_tcolc_eatm_sigma_1 0.00274559226457
158 nm_intp_sigma_pwat_eatm_sigma_4 0.00274237390192
206 nm_intp_uswrf_sfc_m 0.00273603859409
175 nm_intp_sigma_spfh_2m_sigma_1 0.00270191858982
157 nm_intp_sigma_pwat_eatm_sigma_3 0.00269455467991
155 nm_intp_sigma_pwat_eatm_sigma_1 0.00268483618934
153 nm_intp_sigma_ulwrf_tatm_sigma_4 0.00268218132498
229 nm_intp_sigma_dswrf_sfc_sigma_m 0.00262544346982
149 nm_intp_sigma_ulwrf_tatm_sigma_0 0.00259876118532
186 nm_intp_sigma_tmax_2m_sigma_2 0.00258830211405
194 nm_intp_sigma_tmp_2m_sigma_0 0.00257868254301
141 nm_intp_sigma_uswrf_sfc_sigma_2 0.00249846133015
214 nm_intp_tcolc_eatm_m 0.00244735870976
138 nm_intp_sigma_dlwrf_sfc_sigma_4 0.00243241316164
146 nm_intp_sigma_ulwrf_sfc_sigma_2 0.00241182203951
83 nmft_intp_uswrf_sfc/dswrf_sfc_4 0.0023914459365
115 nmft_intp_tmp_2m/tmp_sfc_1 0.00237533229755
192 nm_intp_sigma_tmin_2m_sigma_3 0.00229906422861
179 nm_intp_sigma_tcolc_eatm_sigma_0 0.00227908831304
105 nmft_intp_tmax_2m/tmin_2m_1 0.00222548829424
191 nm_intp_sigma_tmin_2m_sigma_2 0.0022238443713
6 nm_intp_dswrf_sfc_2 0.0021640161251
140 nm_intp_sigma_uswrf_sfc_sigma_1 0.0021373125369
87 nmft_intp_ulwrf_sfc/dlwrf_sfc_3 0.00212547445316
74 nm_intp_tmp_sfc_0 0.00212281138235
109 nmft_intp_tmp_2m-tmp_sfc_0 0.00210867442224
232 nm_intp_sigma_ulwrf_sfc_sigma_m 0.00209959266202
102 nmft_intp_tmax_2m-tmin_2m_3 0.00209653809946
178 nm_intp_sigma_spfh_2m_sigma_4 0.00209484268457
195 nm_intp_sigma_tmp_2m_sigma_1 0.00193482448906
137 nm_intp_sigma_dlwrf_sfc_sigma_3 0.0019295106488
144 nm_intp_sigma_ulwrf_sfc_sigma_0 0.00188975990726
86 nmft_intp_ulwrf_sfc/dlwrf_sfc_2 0.00187698905641
44 nm_intp_pres_msl_0 0.00187414531209
124 nmft_intp_apcp_sfc/pwat_eatm_0 0.00186469216628
95 nmft_intp_dlwrf_sfc/dswrf_sfc_1 0.00185496141397
147 nm_intp_sigma_ulwrf_sfc_sigma_3 0.00182078543725
211 nm_intp_apcp_sfc_m 0.00180084114163
160 nm_intp_sigma_tcdc_eatm_sigma_1 0.00179165215439
197 nm_intp_sigma_tmp_2m_sigma_3 0.00178464617221
99 nmft_intp_tmax_2m-tmin_2m_0 0.00178232106395
168 nm_intp_sigma_apcp_sfc_sigma_4 0.00175585156557
71 nm_intp_tmp_2m_2 0.00173381618697
143 nm_intp_sigma_uswrf_sfc_sigma_4 0.00170983430074
202 nm_intp_sigma_tmp_sfc_sigma_3 0.0017044889173
199 nm_intp_sigma_tmp_sfc_sigma_0 0.00166898005013
242 nm_intp_sigma_tmp_2m_sigma_m 0.00159547494109
228 nmft_intp_apcp_sfc/pwat_eatm_m 0.00155678652092
167 nm_intp_sigma_apcp_sfc_sigma_3 0.00155209946342
132 nm_intp_sigma_dswrf_sfc_sigma_3 0.0015517876562
29 nm_intp_pwat_eatm_0 0.0015330160532
193 nm_intp_sigma_tmin_2m_sigma_4 0.00142793854348
106 nmft_intp_tmax_2m/tmin_2m_2 0.0013981797922
119 nmft_intp_apcp_sfc-pwat_eatm_0 0.00136652313988
14 nm_intp_uswrf_sfc_0 0.00127576622169
203 nm_intp_sigma_tmp_sfc_sigma_4 0.00124725767083
165 nm_intp_sigma_apcp_sfc_sigma_1 0.00124492674509
128 nmft_intp_apcp_sfc/pwat_eatm_4 0.00124057333658
139 nm_intp_sigma_uswrf_sfc_sigma_0 0.00118761829432
150 nm_intp_sigma_ulwrf_tatm_sigma_1 0.00116082118408
121 nmft_intp_apcp_sfc-pwat_eatm_2 0.00115147786458
181 nm_intp_sigma_tcolc_eatm_sigma_2 0.00115052793417
79 nmft_intp_uswrf_sfc/dswrf_sfc_0 0.00114947912617
185 nm_intp_sigma_tmax_2m_sigma_1 0.0011332881931
136 nm_intp_sigma_dlwrf_sfc_sigma_2 0.00111675523466
4 nm_intp_dswrf_sfc_0 0.00102285337821
108 nmft_intp_tmax_2m/tmin_2m_4 0.000984511333671
112 nmft_intp_tmp_2m-tmp_sfc_3 0.00097912201755
154 nm_intp_sigma_pwat_eatm_sigma_0 0.000968809366602
59 nm_intp_tmax_2m_0 0.000942639829353
142 nm_intp_sigma_uswrf_sfc_sigma_3 0.000929053659837
151 nm_intp_sigma_ulwrf_tatm_sigma_2 0.000927726471837
219 nmft_intp_uswrf_sfc/dswrf_sfc_m 0.00089965037255
114 nmft_intp_tmp_2m/tmp_sfc_0 0.000889367989948
26 nm_intp_ulwrf_tatm_2 0.000884318432684
55 nm_intp_tcolc_eatm_1 0.000792314250232
131 nm_intp_sigma_dswrf_sfc_sigma_2 0.000774750463857
236 nm_intp_sigma_apcp_sfc_sigma_m 0.00074263113511
98 nmft_intp_dlwrf_sfc/dswrf_sfc_4 0.000701295129819
169 nm_intp_sigma_pres_msl_sigma_0 0.00061794327581
135 nm_intp_sigma_dlwrf_sfc_sigma_1 0.000603512211269
56 nm_intp_tcolc_eatm_2 0.000575352877854
166 nm_intp_sigma_apcp_sfc_sigma_2 0.000554729758223
129 nm_intp_sigma_dswrf_sfc_sigma_0 0.0005224226521
89 nmft_intp_ulwrf_sfc/uswrf_sfc_0 0.000510525993389
101 nmft_intp_tmax_2m-tmin_2m_2 0.000497638535425
122 nmft_intp_apcp_sfc-pwat_eatm_3 0.000463027506918
11 nm_intp_dlwrf_sfc_2 0.00045730955777
88 nmft_intp_ulwrf_sfc/dlwrf_sfc_4 0.000400631585475
184 nm_intp_sigma_tmax_2m_sigma_0 0.000396263578265
10 nm_intp_dlwrf_sfc_1 0.000372609065828
196 nm_intp_sigma_tmp_2m_sigma_2 0.000353331897929
111 nmft_intp_tmp_2m-tmp_sfc_2 0.000329557998765
41 nm_intp_apcp_sfc_2 0.000318739593537
118 nmft_intp_tmp_2m/tmp_sfc_4 0.000198662838212
92 nmft_intp_ulwrf_sfc/uswrf_sfc_3 0.000172256876068
91 nmft_intp_ulwrf_sfc/uswrf_sfc_2 8.04119652824e-05



16.10.2013
^^^^^^^^^^

HK15 experiment - same as HK10 but with interp6 (spline) instead of interp5 (GP).

Here are the TT results for HK10 and HK14

HK14

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=5,
             min_samples_split=2, n_estimators=2000, random_state=2,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=33,
       est__min_samples_leaf=5, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=2, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_mask=False, with_solar=False,
       with_stationinfo=True)

Without masking
MAE:  1960357.90
RMSE: 3065497.83
R2: 0.84

With masking
MAE:  1913991.42
RMSE: 2947062.05
R2: 0.85

LB: 1938378.35


HK10

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=33,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_mask=True, with_solar=True,
       with_stationinfo=True)

Without masking
MAE:  1970118.02
RMSE: 3078680.34
R2: 0.84

With masking
MAE:  1923963.48
RMSE: 2960936.41
R2: 0.85

LB: 1929699.30


------------------

Grid (interp6):

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.022, loss=lad,
             max_depth=6, max_features=36, min_samples_leaf=3,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.022,
       est__loss=lad, est__max_depth=6, est__max_features=36,
       est__min_samples_leaf=3, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_mask=False, with_solar=False,
       with_stationinfo=True)

Without masking
MAE:  1958663.06
RMSE: 3063509.66
R2: 0.84

With masking
MAE:  1912385.51
RMSE: 2945357.17
R2: 0.85



-------------------

interp6

EnsembleKrigingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02,
loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=9,
             min_samples_split=2, n_estimators=1000, random_state=1,
             subsample=1.0, verbose=1),
           est__alpha=0.9, est__init=None, est__learning_rate=0.02,
           est__loss=lad, est__max_depth=6, est__max_features=33,
           est__min_samples_leaf=9, est__min_samples_split=2,
           est__n_estimators=1000, est__random_state=1, est__subsample=1.0,
           est__verbose=1, intp_blocks=('nm_intp', 'nmft_intp'),
           with_date=True, with_mask=False, with_solar=False,
           with_stationinfo=True)

Without masking
MAE:  1964788.67
RMSE: 3072029.52
R2: 0.84

With masking
MAE:  1918357.25
RMSE: 2954007.09


________________________________________________________________________________

KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.02, loss=lad,
             max_depth=6, max_features=33, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.02,
       est__loss=lad, est__max_depth=6, est__max_features=33,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_mask=True, with_solar=True,
       with_stationinfo=True)
--data=data/interp9_data.pkl

MAE:  1969303.48
RMSE: 3081565.70
R2: 0.84

________________________________________________________________________________
With masking
MAE:  1922987.14
RMSE: 2963366.83
R2: 0.85



KringingModel(est=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.023, loss=lad,
             max_depth=6, max_features=30, min_samples_leaf=9,
             min_samples_split=2, n_estimators=2000, random_state=1,
             subsample=1.0, verbose=1),
       est__alpha=0.9, est__init=None, est__learning_rate=0.023,
       est__loss=lad, est__max_depth=6, est__max_features=30,
       est__min_samples_leaf=9, est__min_samples_split=2,
       est__n_estimators=2000, est__random_state=1, est__subsample=1.0,
       est__verbose=1,
       intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
       with_date=True, with_mask=True, with_solar=True,
       with_stationinfo=True)

Without masking
MAE:  1880074.27
RMSE: 2931100.49
R2: 0.85

________________________________________________________________________________
With masking
MAE:  1834621.03
RMSE: 2808696.40
R2: 0.86
