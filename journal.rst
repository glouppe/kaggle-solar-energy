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
