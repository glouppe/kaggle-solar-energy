=======
Solaris
=======

Requirements
============

pandas>=0.12.0
numpy>=1.7.1
docopt>=0.6.1

Data
====

All data files are expected to be found in the `data` subfolder.
Each dataset is a joblib pickled dict holding training and test
set as ``solaris.sa.StructuredArray``.

To create this format from the Kaggle download use the script
``solaris.load_data``::

    $ python -m solaris.load_data

This will save the transformed data into the `data` subfolder.
The name is `data/data.pkl`.

To generate the interpolated blocks use the script `solaris.kringing`.

    $ python -m solaris.kringing
