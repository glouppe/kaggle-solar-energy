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
