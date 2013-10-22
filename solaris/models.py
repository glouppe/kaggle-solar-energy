import numpy as np
import pandas as pd
import gc
import IPython

from time import time
from scipy import ndimage
from collections import OrderedDict

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from . import util
from .sa import StructuredArray


class ValueTransformer(BaseEstimator, TransformerMixin):
    """Transforms StructuredArray to numpy array. """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.values()
        print('transform X to shape %s' % str(X.shape))
        return X


class ResampleTransformer(BaseEstimator, TransformerMixin):

    zoom = 0.25
    order = 0
    flatten = True

    def fit(self, X, y=None):
        return self

    def transform(self, X_st, y=None):
        X_nm = X_st.nm[:, 0]
        # mean ensemble
        X_nm = np.mean(X_nm, axis=1)
        # mean hour
        X_nm = np.mean(X_nm, axis=1)
        # X_nm.shape = (n_days, n_lat, l_lon)
        print X_nm.shape

        n_days = X_nm.shape[0]
        out = None
        for i in range(n_days):
            resampled_day = ndimage.interpolation.zoom(X_nm[i], self.zoom)
            if out is None:
                out = np.zeros((n_days,) + resampled_day.shape,
                               dtype=np.float32)
            out[i] = resampled_day
        if self.flatten:
            out = out.reshape(out.shape[0], np.prod(out.shape[1:]))
        X_st['nm_res'] = out
        return X_st


class FunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, block='nm', ops=(
        ('tcolc_eatm', 'pow', 0.15),
        ('apcp_sfc', 'pow', 0.15),
        ('uswrf_sfc', '/', 'dswrf_sfc'),
        ('ulwrf_sfc', '/', 'dlwrf_sfc'),
        ('ulwrf_sfc', '/', 'uswrf_sfc'),
        ('dlwrf_sfc', '/', 'dswrf_sfc'),
        ('tmax_2m', '-', 'tmin_2m'),
        ('tmp_2m', '-', 'tmp_sfc'),
        ('apcp_sfc', '-', 'pwat_eatm'),
        ('apcp_sfc', '/', 'pwat_eatm'),
        ), new_block='nm_trans'):
        """Apply func transformation to X creating a new block.

        Parameters
        ----------
        block : str
            The name of the block to which ``ops`` are applied.
        new_block : str
            The name of the new block holding the result of ``ops``.
        ops : seq of (left, op, right)
            The sequence of operations to be applied to ``block``.
        """
        self.block = block
        self.ops = ops
        self.new_block = new_block

    def fit(self, *args, **kw):
        return self

    # @profile
    def transform(self, X, y=None, dates=None):
        block_name = self.block
        block_fx_name = []
        block_shape = X[self.block].shape
        print('[FT] nr new features: %d' % len(self.ops))
        print self.ops
        out = np.zeros((block_shape[0], len(self.ops)) +
                        block_shape[2:], dtype=np.float32)
        for op_idx, (left, op, right) in enumerate(self.ops):
            if op == 'pow':
                i = X.fx_name[block_name].index(left)
                vals = X[block_name][:, i]
                vals = vals ** right
            elif op == '/':
                i = X.fx_name[block_name].index(left)
                j = X.fx_name[block_name].index(right)
                vals = (X[block_name][:, i] /
                        X[block_name][:, j])
                # might be infs in there
                # vals[~np.isfinite(vals)] = 0.0
                # if both are zero then 0 else large number
                mask_j = X[block_name][:, j] == 0.0
                mask_i = X[block_name][:, i] == 0.0
                vals[np.logical_and(mask_i, mask_j)] = 0.0
                vals[np.logical_and(~mask_i, mask_j)] = 1e15

            elif op == '-':
                i = X.fx_name[block_name].index(left)
                j = X.fx_name[block_name].index(right)
                vals = (X[block_name][:, i] -
                        X[block_name][:, j])

            out[:, op_idx] = vals
            block_fx_name.append(''.join(map(str, [left, op, right])))

        X[self.new_block] = out
        X.fx_name[self.new_block] = block_fx_name
        if hasattr(X, 'fx_name'):
            X.fx_name[self.new_block] = block_fx_name
        if y is not None:
            return X, y
        else:
            return X


class DateTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, op='doy'):
        self.op = op

    def fit(self, X, y=None):
        if self.op == 'month' and hasattr(X, 'date'):
            month = X.date.map(lambda x: x.month)
            self.lb = LabelBinarizer()
            self.lb.fit(month)
        return self

    # @profile
    def transform(self, X, y=None):
        if hasattr(X, 'date'):
            date = X.date

            year = date.map(lambda x: x.year)
            year = year.astype(np.float32)
            year = year.reshape((year.shape[0], 1))
            X['year'] = year
            X.fx_name['year'] = 'year'

            op = self.op
            if op == 'doy':
                vals = date.map(lambda x: x.dayofyear)
            elif op == 'center':
                doy = date.map(lambda x: x.dayofyear)
                vals = np.abs(doy - (365.25 / 2.0))
            elif op == 'month':
                month = date.map(lambda x: x.month)
                vals = self.lb.transform(month)

            vals = vals.astype(np.float32)
            if vals.ndim == 1:
                vals = vals.reshape((vals.shape[0], 1))

            X['date'] = vals
            if op == 'month':
                X.fx_name['date'] = self.lb.classes_.tolist()
            else:
                X.fx_name['date'] = self.op

        return X


class SolarTransformer(BaseEstimator, TransformerMixin):
    """Compute varios solar features (sun rise, azimuth). """

    def __init__(self, ops=None):
        self.ops = ops

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        from . import sun
        jds = X.date.map(sun.get_jd2)

        sun_riseset = sun.vec_sunriseset(jds, X.station_info[:, :2])
        X['sun_riseset'] = sun_riseset
        X.fx_name['sun_riseset'] = ['sunrise', 'sunset']

        # sun_noon = sun.vec_solnoon(jds, X.station_info[:, 1])
        # X['sun_noon'] = sun_noon[:, np.newaxis]
        # X.fx_name['sun_noon'] = ['sun_noon']

        # stuff, stuff_names = sun.calc_solar_stuff(jds)
        # X['sun_stuff'] = stuff
        # X.fx_name['sun_stuff'] = stuff_names

        return X


class DelBlockTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, rm_lst=None):
        self.rm_lst = rm_lst

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **kw):
        for b_name in self.rm_lst:
            del X.blocks[b_name]
        return X


class BaselineTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, b_name='nm'):
        self.b_name = b_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_nm = X[self.b_name]
        print('[baseline tf] old shape: %s' % str(X_nm.shape))
        # mean over ensembles
        X_nm = np.mean(X_nm, axis=2)
        # mean over hours
        X_nm = np.mean(X_nm, axis=2)
        # reshape
        X_nm = X_nm.reshape(X_nm.shape[0], np.prod(X_nm.shape[1:]))

        X[self.b_name] = X_nm
        print('[baseline tf] new shape: %s' % str(X_nm.shape))
        return X


class PipelineModel(BaseEstimator, RegressorMixin):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y, **kw):
        if y.shape[1] == 1:
            y = y.ravel()
        self.pipeline.fit(X, y)
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        pred = self.pipeline.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape((pred.shape[0], 1))
        return pred

    def transform(self, X, y):
        return self.pipeline.transform(X), y


class Baseline(PipelineModel):

    def __init__(self, est=None, date='center'):
        self.est = est
        self.date = date
        steps = [
            ('ft', FunctionTransformer(block='nm', new_block='nmft',
                                            ops=(
                                                ('uswrf_sfc', '/', 'dswrf_sfc'),
                                                ('ulwrf_sfc', '/', 'dlwrf_sfc'),
                                                ('ulwrf_sfc', '/', 'uswrf_sfc'),
                                                ('dlwrf_sfc', '/', 'dswrf_sfc'),
                                                #('dswrf_sfc', 'pow', 1.0),
                                                ))),
            ('bl_trans', BaselineTransformer(b_name='nm')),
            ('bl_trans2', BaselineTransformer(b_name='nmft')),
            ]
        if self.date:
            steps.append(('date', DateTransformer(op=self.date)))

        steps.append(('vals', ValueTransformer()))
        if self.est is not None:
            steps.append(('est', est))

        self.pipeline = Pipeline(steps)


class IndividualEstimator(BaseEstimator, RegressorMixin):

    def __init__(self, base_est):
        self.base_est = base_est

    def fit(self, X, y):
        self.n_outputs_ = y.shape[1]
        self.estimators_ = []
        t0 = time()
        for i in range(self.n_outputs_):
            print 'fitting estimator %d' % i
            est = clone(self.base_est)
            est.fit(X, y[:, i])
            self.estimators_.append(est)
        print('fitted %d estimators in %ds' % (i, time() - t0))
        return self

    def predict(self, X):
        pred = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        for i in range(self.n_outputs_):
            pred[:, i] = self.estimators_[i].predict(X)
        return pred


class KringingModel(BaseEstimator, RegressorMixin):
    """Assumes that blocks ``nm_intp`` is present - interpolation
    of station parameters by Kriging (see solaris.kringing).

    Attributes
    ----------
    ens_mean : bool
        Whether or not ens is included in the ``nm_inpt`` feature block.
        If True it will create mean and std features.

    Parameters
    ----------
    intp_blocks : seq of str
        The blocks of X that are used to create the features; defaults
        to interpolated parameters, functional transformation, and
        standard deviation of intp.
    with_stationinfo : bool
        Whether or not to include station lat, lon, and elev
    with_date : bool
        Whether or not to include day of year feature.
    with_solar : bool
        Whether or not to include time of sunrise, sunset and diff thereof.
    with_mask : bool
        Whether to exclude samples that have been imputed by the organizers
        (see utils.mask_missing_values).

    """

    ens_mean = False

    def __init__(self, est, intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
                 with_stationinfo=True, with_date=True,
                 with_solar=False, with_mask=False,
                 with_climate=False,
                 with_mean_history=False,
                 with_history=False,
                 with_mask_hard=False):
        self.est = est
        self.intp_blocks = intp_blocks
        self.with_date = with_date
        self.with_solar = with_solar
        self.with_mask = with_mask
        self.with_stationinfo = with_stationinfo
        self.with_climate = with_climate
        self.with_mean_history = with_mean_history
        self.with_history = with_history
        self.with_mask_hard = with_mask_hard

    def fit(self, X_st, y):
        self.n_stations = y.shape[1]

        mask = None
        if self.with_mask:
            mask = util.clean_missing_labels(y)

        if self.with_mask_hard:
            print('|mask|: %d' % mask.sum())
            hard = np.load('data/hard_train_05.npy')
            if hard.shape[0] < X_st.shape[0]:
                # for submit runs - FIXME will break if TT split not 0.5
                hard_2 = np.load('data/hard_test_05.npy')
                hard = np.r_[hard, hard_2]
            print('|hard|: %d' % hard.sum())
            mask = np.logical_or(mask, hard)
            print('|mask|: %d' % mask.sum())

        X = self.transform(X_st)
        y = self.transform_labels(y)

        if mask is not None:
            mask = self.transform_labels(mask)
            print('remove masked samples: %d' % mask.sum())
            y = y[~mask]
            X = X[~mask]

        self.est.fit(X, y)
        return self

    def predict(self, X_st, y_test=None):
        X = self.transform(X_st)

        pred = self.est.predict(X)
        pred = pred.reshape((X_st.shape[0], self.n_stations))
        return pred

    def transform_labels(self, y):
        return y.ravel()

    def transform(self, X_st):
        out = []
        fx_names = []
        n_days = X_st.shape[0]

        orig_date = X_st.date.copy()

        ## FIXME forgot feature names for nm_intp
        if 'nm_intp' not in X_st.fx_name:
            X_st.fx_name['nm_intp'] = X_st.fx_name['nm']
        if 'nm_intp_sigma' not in X_st.fx_name:
            X_st.fx_name['nm_intp_sigma'] = map(lambda s: '%s_sigma' % s,
                                                X_st.fx_name['nm'])

        if self.ens_mean:
            # mean and std
            X_nm_intp = X_st['nm_intp']
            X_st['nm_intp'] = X_nm_intp.mean(axis=2)
            X_st['nm_intp_sigma'] = X_nm_intp.std(axis=2)

        ## transform function
        ft = FunctionTransformer(block='nm_intp', new_block='nmft_intp',
                                 ops=(
                                     ('uswrf_sfc', '/', 'dswrf_sfc'),
                                     ('ulwrf_sfc', '/', 'dlwrf_sfc'),
                                     ('ulwrf_sfc', '/', 'uswrf_sfc'),
                                     ('dlwrf_sfc', '/', 'dswrf_sfc'),
                                     ('tmax_2m', '-', 'tmin_2m'),
                                     ('tmax_2m', '/', 'tmin_2m'),
                                     ('tmp_2m', '-', 'tmp_sfc'),
                                     ('tmp_2m', '/', 'tmp_sfc'),
                                     ('apcp_sfc', '-', 'pwat_eatm'),
                                     ('apcp_sfc', '/', 'pwat_eatm'),

                                     ('uswrf_sfc', '/', 'dlwrf_sfc'),
                                     ('ulwrf_sfc', '/', 'dswrf_sfc'),
                                     ))
        X_st = ft.transform(X_st)

        if self.with_solar:
            solar_tf = SolarTransformer()
            X_st = solar_tf.transform(X_st)

            # sun rise sun set
            sun_riseset = X_st['sun_riseset']
            sun_riseset = sun_riseset.reshape((n_days * self.n_stations, 2))

            #out.append(sun_riseset)
            #fx_names.extend(X_st.fx_name['sun_riseset'])
            out.append((sun_riseset[:, 1] - sun_riseset[:, 0])[:, np.newaxis])
            fx_names.append('delta_sunrise_sunset')

            # sun noon
            # sun_noon = X_st['sun_noon']
            # sun_noon = sun_noon.reshape((n_days * self.n_stations, 1))
            # out.append(sun_noon)
            # fx_names.extend(X_st.fx_name['sun_noon'])

            # stuff = X_st['sun_stuff']
            # stuff = np.repeat(stuff, self.n_stations, axis=0)
            # out.append(stuff[:, np.newaxis])
            # print out[-1].shape
            # fx_names.extend(X_st.fx_name['sun_stuff'])

        ## transform date
        if self.with_date:
            date_tf = DateTransformer('doy')
            X_st = date_tf.transform(X_st)
            date = np.repeat(X_st.date, self.n_stations)[:, np.newaxis]
            out.append(date.astype(np.float32))
            fx_names.append('doy')

            # FIXME added year
            # year = np.repeat(X_st.year, self.n_stations)[:, np.newaxis]
            # out.append(year.astype(np.float32))
            # fx_names.append('year')

        # ## transform station-info
        if self.with_stationinfo:
            stinfo = X_st.station_info
            # add station id as col
            #stinfo = np.c_[stinfo, np.arange(X_st.station_info.shape[0])]
            stinfo = np.tile(stinfo, (n_days, 1))
            out.append(stinfo.astype(np.float32))
            fx_names.extend(['lat', 'lon', 'elev'])

        ## transform nm_intp to hourly features
        for b_name in self.intp_blocks:
            X = X_st[b_name]
            print b_name, X.shape
            # swap features and station interpolations
            X = np.swapaxes(X, 1, 3)
            fx_names.extend(['_'.join((b_name, fx, str(h)))
                             for fx in X_st.fx_name[b_name]
                             for h in range(X.shape[-2])])
            X = X.reshape((np.prod(X.shape[:2]), -1)).astype(np.float32)
            out.append(X)

            #history
            if self.with_history:
                X = X.copy()
                X[1:] = X[:-1]
                out.append(X)
                fx_names.extend(['_'.join((b_name, fx, str(h), 't-1'))
                             for fx in X_st.fx_name[b_name]
                             for h in range(X.shape[-2])])

        ## transform to mean features
        for b_name in self.intp_blocks:
            X = X_st[b_name]
            print b_name, X.shape
            # swap features and station interpolations
            X = np.swapaxes(X, 1, 3)
            # n_day x n_stat x n_hour x n_fx
            X = np.mean(X, axis=2)
            fx_names.extend(['_'.join((b_name, fx, 'm'))
                             for fx in X_st.fx_name[b_name]])
            X = X.reshape((np.prod(X.shape[:2]), -1)).astype(np.float32)
            out.append(X)

            # history
            if self.with_mean_history:
                X = X.copy()
                X[1:] = X[:-1]
                out.append(X)
                fx_names.extend(['_'.join((b_name, fx, 'm_t-1'))
                                 for fx in X_st.fx_name[b_name]])


        if self.with_climate:
            date = orig_date
            month = date.map(lambda d: d.month).astype(np.float32)
            month = np.repeat(month, self.n_stations)[:, np.newaxis]
            stid = np.tile(np.arange(self.n_stations), n_days)[:, np.newaxis]
            out.extend((month, stid))
            fx_names.extend(['month', 'stid'])

        for o in out:
            print o.shape
        out = np.hstack(out)
        self.fx_names_ = fx_names
        print('transform to shape: %s' % str(out.shape))
        print('dtype: %s' % out.dtype)
        print('size of X in mb: %.2f' % (out.nbytes / 1024.0 / 1024.0))
        return out


class ClimateEstimator(BaseEstimator):

    def fit(self, X, y):
        #assume last two fx are month and stid
        df = pd.DataFrame(data={'month': X[:, -2].astype(np.int) - 1,
                                'stid': X[:, -1].astype(np.int),
                                'y': y})
        self.climate = df.groupby(('month', 'stid')).y.mean().reshape((12, 98))
        #IPython.embed()
        return self

    def predict(self, X):
        month = X[:, -2].astype(np.int) - 1
        stid = X[:, -1].astype(np.int)
        y = self.climate[month, stid][:, np.newaxis]
        return y


class PertubedKrigingModel(KringingModel):

    ens_mean = False
    n_pertubations = 4

    def fit(self, X_st, y):
        y = np.tile(y, (1, self.n_pertubations))
        return super(PertubedKrigingModel, self).fit(X_st, y)

    def predict(self, X_st):
        pred = super(PertubedKrigingModel, self).predict(X_st)
        n_stations = self.n_stations / self.n_pertubations
        pred = pred.reshape((X_st.shape[0], self.n_pertubations, n_stations))
        pred = pred.mean(axis=1)
        return pred


class PertubedPredictionKrigingModel(KringingModel):

    ens_mean = False
    n_pertubations = 4

    def fit(self, X_st, y):
        n_stations = X_st.nm_intp.shape[-1]
        nm_intp = X_st.nm_intp[:, :, :, :98]
        nm_intp_sigma = X_st.nm_intp_sigma[:, :, :, :98]
        X_st['nm_intp'] = nm_intp
        X_st['nm_intp_sigma'] = nm_intp_sigma
        X_st.station_info = X_st.station_info[:98]
        super(PertubedPredictionKrigingModel, self).fit(X_st, y)
        self.n_stations = n_stations

    def predict(self, X_st):
        pred = super(PertubedPredictionKrigingModel, self).predict(X_st)
        n_stations = self.n_stations / self.n_pertubations
        pred = pred.reshape((X_st.shape[0], self.n_pertubations, n_stations))
        pred = np.mean(pred, axis=1)
        return pred


class EnsembleKrigingModel(KringingModel):

    ens_mean = False

    def transform_labels(self, y, n_ens):
        y = np.tile(y, n_ens)
        return y.ravel()

    def transform(self, X):
        nm_intp = X.nm_intp
        nm_intp = nm_intp.swapaxes(1, 2)
        n_days = nm_intp.shape[0]
        n_ens = nm_intp.shape[1]
        nm_intp = nm_intp.reshape((n_days * n_ens, ) +
                                  nm_intp.shape[2:])

        date = np.tile(X.date, n_ens)
        station_info = X.station_info
        fx_name = X.fx_name
        lat = X.lat
        lon = X.lon
        X = StructuredArray(OrderedDict(nm_intp=nm_intp, date=date))
        X.station_info = station_info
        X.fx_name = fx_name
        X.lat = lat
        X.lon = lon
        X = super(EnsembleKrigingModel, self).transform(X)
        return X

    def fit(self, X_st, y):
        self.n_stations = y.shape[1]
        self.n_ens = X_st.nm_intp.shape[2]
        print('n_nes = %d' % self.n_ens)

        mask = None
        if self.with_mask:
            mask = util.clean_missing_labels(y)

        X = self.transform(X_st)
        y = self.transform_labels(y, self.n_ens)
        assert X.shape[0] == y.shape[0]

        if mask is not None:
            mask = self.transform_labels(mask)
            print('remove masked samples: %d' % mask.sum())
            y = y[~mask]
            X = X[~mask]

        print('_' * 80)
        print('fitting estimator...')
        print
        self.est.fit(X, y)
        return self

    def predict(self, X):
        n_days = X.shape[0]
        X = self.transform(X)
        pred = self.est.predict(X)
        print pred.shape
        pred = pred.reshape((n_days, self.n_ens, self.n_stations))
        pred = pred.mean(axis=1)
        return pred


class LocalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, k=1, fxs=None, hour_mean=True, aux=True, ens_std=False,
                 hour_std=False, stid_enc=False):
        self.k = k
        self.hour_mean = hour_mean
        self.fxs = fxs
        self.aux = aux
        self.ens_std = ens_std
        self.hour_std = hour_std
        self.stid_enc = stid_enc

    def fit(self, X, y=None):
        assert y is not None
        n_stations = y.shape[1]
        self.n_stations = n_stations
        assert X.station_info.shape[0] == self.n_stations

        stid = np.arange(98)
        self.stid_lb = LabelBinarizer()
        self.stid_lb.fit(stid)
        return self

    @classmethod
    def transform_labels(cls, y):
        y = y.ravel(1)
        return y

    # @profile
    def transform(self, X, y=None):
        k = self.k
        n_days = X.shape[0]

        ll_coord = X.station_info[:, 0:2]
        lat_idx = np.searchsorted(X.lat, ll_coord[:, 0])
        lon_idx = np.searchsorted(X.lon, ll_coord[:, 1] + 360)

        #IPython.embed()

        n_fx = 0
        for b_name, X_b in X.blocks.iteritems():
            old_n_fx = n_fx
            if self.fxs is not None and b_name not in self.fxs:
                continue
            if X_b.ndim == 6:
                if self.fxs is not None and b_name in self.fxs:
                    n_fxs = len(self.fxs[b_name])
                else:
                    n_fxs = X_b.shape[1]
                shapes = [n_fxs]
                if not self.hour_mean:
                    shapes.append(X_b.shape[3])
                shapes.extend([k * 2, k * 2])
                print b_name, shapes
                n_fx += np.prod(shapes)
            elif X_b.ndim == 1:
                n_fx += 1
            elif X_b.ndim == 2:
                n_fx += X_b.shape[1]
            else:
                raise ValueError('%s has wrong dim: %d' % (b_name, X_b.ndim))
            print 'block: %s as %d n_fx' % (b_name, n_fx - old_n_fx)

        if self.stid_enc:
            n_fx += len(self.stid_lb.classes_)

        # num of features - based on blocks + station info (5 fx)
        if self.aux:
            n_fx = n_fx + 3 + 2 + 2

        X_p = np.zeros((n_days * self.n_stations, n_fx), dtype=np.float32)
        offset = 0

        for bid, b_name in enumerate(X.blocks):
            if self.fxs is not None and b_name not in self.fxs:
                continue
            print 'localizing block: %s' % b_name
            X_b = X[b_name]

            # select fx if fxs given
            if self.fxs is not None and self.fxs.get(b_name, None):
                fxs = self.fxs[b_name]
                idx = [i for i, name in enumerate(X.fx_name[b_name])
                       if name in fxs]
                X_b = X_b[:, idx]

            if X_b.ndim == 6:
                # FIXME over hours
                if self.hour_mean:
                    X_b = np.mean(X_b, axis=3)
                elif self.hour_std:
                    X_b = np.std(X_b, axis=3)

                # over ensembles
                if self.ens_std:
                    X_b = np.std(X_b, axis=2)
                else:
                    X_b = np.mean(X_b, axis=2)

                offset_inc = 0
                for i in range(self.n_stations):
                    lai, loi = lat_idx[i], lon_idx[i]
                    if (self.hour_mean or self.hour_std):
                        blk = X_b[:, :, lai - k:lai + k,
                                  loi - k: loi + k]
                    else:
                        blk = X_b[:, :, :, lai - k:lai + k,
                                  loi - k: loi + k]
                    blk = blk.reshape((blk.shape[0], np.prod(blk.shape[1:])))
                    X_p[i * n_days:((i+1) * n_days),
                        offset:(offset + blk.shape[1])] = blk
                    if i == 0:
                        offset_inc = blk.shape[1]
                    del blk
                    gc.collect()

                offset += offset_inc

            elif X_b.ndim == 1 or (X_b.ndim == 2 and X_b.shape[1] == 1):
                X_p[:, offset:offset + 1] = np.tile(X_b.ravel(),
                                                    self.n_stations)[:, np.newaxis]
                offset += 1
            elif X_b.ndim == 2:
                # FIXME wrong stitching together stuff
                print('block: %s will be repeated for each station' % b_name)
                width = X_b.shape[1]
                X_p[:, offset:offset + width] = np.tile(X_b, (self.n_stations, 1))
                #IPython.embed()
                offset += width
            else:
                raise ValueError('%s has wrong dim: %d' % (b_name, X_b.ndim))

        if self.stid_enc:
            stid = np.repeat(self.stid_lb.classes_, n_days)
            stid_enc = self.stid_lb.transform(stid)
            X_p[:, offset:(offset + stid_enc.shape[1])] = stid_enc
            offset += stid_enc.shape[1]

        if self.aux:
            # lat, lon, elev
            X_p[:, offset:(offset + 3)] = np.repeat(X.station_info, n_days, axis=0)
            offset += 3

            # compute pos of station within grid cell (in degree lat lon)
            lat_idx = np.repeat(lat_idx, n_days)
            lon_idx = np.repeat(lon_idx, n_days)
            # offset - 3 is station lat
            X_p[:, offset] = (X_p[:, offset - 3] - X.lat[lat_idx])
            # offset - 2 is station lon
            X_p[:, offset + 1] = (X_p[:, offset - 2] - (X.lon[lon_idx] - 360.))

            # FIXME add lat lon idx
            offset += 2
            X_p[:, offset] = lat_idx
            X_p[:, offset + 1] = lon_idx

        print 'X_p.shape: ', X_p.shape
        return X_p


class LocalModel(BaseEstimator, RegressorMixin):

    def __init__(self, est, clip=False):
        self.est = est
        self.clip=clip
        steps = [
            ('date', DateTransformer(op='doy')),
            # ('ft', FunctionTransformer(block='nm', new_block='nmft',
            #                            ops=(
            #                                ('uswrf_sfc', '/', 'dswrf_sfc'),
            #                                ('ulwrf_sfc', '/', 'dlwrf_sfc'),
            #                                ('ulwrf_sfc', '/', 'uswrf_sfc'),
            #                                ('dlwrf_sfc', '/', 'dswrf_sfc'),
            #                                ('tmax_2m', '-', 'tmin_2m'),
            #                                ('tmax_2m', '/', 'tmin_2m'),
            #                                ('tmp_2m', '-', 'tmp_sfc'),
            #                                ('tmp_2m', '/', 'tmp_sfc'),
            #                                ('apcp_sfc', '-', 'pwat_eatm'),
            #                                ('apcp_sfc', '/', 'pwat_eatm'),

            #                                ('uswrf_sfc', '/', 'dlwrf_sfc'),
            #                                ('ulwrf_sfc', '/', 'dswrf_sfc'),
            #                                     ))),

                 ]
        self.pipeline = Pipeline(steps)

        l1 = LocalTransformer(hour_mean=True, k=1, aux=True)
        #l2 = LocalTransformer(hour_mean=False, ens_std=True, k=1)

        self.fu = FeatureUnion([('hm_k1', l1),
                                #('hs_k1', l2),
                                ])

    def transform(self, X, y):
        self.n_stations = y.shape[1]
        ## FIXME hack
        X = self.pipeline.transform(X)
        self.fu.fit(X, y)
        X = self.fu.transform(X)
        y = LocalTransformer.transform_labels(y)
        return X, y

    def fit(self, X, y, **kw):
        self.n_stations = y.shape[1]
        X = self.pipeline.transform(X)
        self.fu.fit(X, y)
        X = self.fu.transform(X)
        y = LocalTransformer.transform_labels(y)
        assert X.shape[0] == y.shape[0]
        print 'LocalModel: fit est on X: %s' % str(X.shape)
        t0 = time()
        self.est.fit(X, y)
        print('Est fitted in %.2fm' % ((time() - t0) / 60.))
        if self.clip:
            self.clip_high = y.max()
            self.clip_low = y.min()
        return self

    def predict(self, X):
        n_days = X.shape[0]
        X = self.pipeline.transform(X)
        X = self.fu.transform(X)
        pred = self.est.predict(X)
        print X.shape, pred.shape
        # reshape - first read days (per station)
        pred = pred.reshape((self.n_stations, n_days)).T
        if self.clip:
            pred = np.clip(pred, self.clip_low, self.clip_high)
        return pred


class EdgeLocal(BaseEstimator, RegressorMixin):

    def __init__(self, est, block_names=('nm', 'nmft'), clip=False):
        self.est = est
        self.block_names = block_names
        self.clip=clip
        steps = [
            ('date', DateTransformer(op='doy')),
            ('ft', FunctionTransformer(block='nm', new_block='nmft',
                                       ops=(
                                           ('uswrf_sfc', '/', 'dswrf_sfc'),
                                           ('ulwrf_sfc', '/', 'dlwrf_sfc'),
                                           ('ulwrf_sfc', '/', 'uswrf_sfc'),
                                           ('dlwrf_sfc', '/', 'dswrf_sfc'),
                                           ('tmax_2m', '-', 'tmin_2m'),
                                           ('tmax_2m', '/', 'tmin_2m'),
                                           ('tmp_2m', '-', 'tmp_sfc'),
                                           ('tmp_2m', '/', 'tmp_sfc'),
                                           ('apcp_sfc', '-', 'pwat_eatm'),
                                           ('apcp_sfc', '/', 'pwat_eatm'),

                                           ('uswrf_sfc', '/', 'dlwrf_sfc'),
                                           ('ulwrf_sfc', '/', 'dswrf_sfc'),
                                                ))),
                 ]
        self.pipeline = Pipeline(steps)

    def _load_grid_elev(self):
        import netCDF4
        ds = netCDF4.Dataset('data/gefs_elevations.nc')
        return ds.variables['elevation_control'][:]

    def _transform(self, X):
        n_days = X.shape[0]
        n_stations = X.station_info.shape[0]

        # compute coordinates
        ll_coord = X.station_info[:, 0:2]  # lat-lon
        lat_idx = np.searchsorted(X.lat, ll_coord[:, 0]).repeat(4)
        lon_idx = np.searchsorted(X.lon, ll_coord[:, 1] + 360).repeat(4)
        ll_coord = ll_coord.repeat(4, axis=0)

        lat_offsets = np.tile(np.array([0, 0, -1, -1]), n_stations)
        lon_offsets = np.tile(np.array([0, -1, 0, -1]), n_stations)

        grid_coord = np.c_[X.lat[lat_idx + lat_offsets],
                           X.lon[lon_idx + lon_offsets] - 360.]
        grid_elev = self._load_grid_elev()[lat_idx + lat_offsets,
                                           lon_idx + lon_offsets][:, np.newaxis]
        rel_coord = ll_coord - grid_coord

        # coordindate and date features
        rel_coord = np.tile(rel_coord, (n_days, 1)).astype(np.float32)
        abs_coord = np.tile(ll_coord, (n_days, 1)).astype(np.float32)
        elev_coord = np.tile(X.station_info[:, 2].repeat(4)[:, np.newaxis],
                             (n_days, 1)).astype(np.float32)
        doy = X.date.repeat(n_stations * 4)[:, np.newaxis].astype(np.float32)

        grid_coord = np.tile(grid_coord, (n_days, 1)).astype(np.float32)
        grid_elev = np.tile(grid_elev, (n_days, 1)).astype(np.float32)
        blocks = [grid_coord, rel_coord, abs_coord, elev_coord,
                  grid_elev, doy]

        # FIXME add fx_names

        for block_name in self.block_names:
            nm_mean = X[block_name].mean(axis=2).astype(np.float32)
            nm_std = X[block_name].std(axis=2).astype(np.float32)
            for nm in (nm_mean, nm_std):
                out = nm[:, :, :, lat_idx + lat_offsets, lon_idx + lon_offsets]
                out= out.swapaxes(1, 3)
                out = out.reshape((out.shape[0] * out.shape[1], -1))
                blocks.append(out)

            # add mean fx
            nm = nm_mean.mean(axis=2)
            out = nm[:, :, lat_idx + lat_offsets, lon_idx + lon_offsets]
            out= out.swapaxes(1, 2)
            out = out.reshape((out.shape[0] * out.shape[1], -1))
            blocks.append(out)

        for b in blocks:
            print b.shape

        out = np.hstack(blocks).astype(np.float32)
        return out

    def transform_labels(self, y):
        """Ravels and copies each value 4 times """
        return y.ravel().repeat(4)

    def transform(self, X):
        X = self.pipeline.transform(X)
        X = self._transform(X)
        return X

    def fit(self, X, y, **kw):
        self.n_stations = y.shape[1]

        X = self.transform(X)
        y = self.transform_labels(y)

        assert X.shape[0] == y.shape[0]
        print 'EdgeLocal: fit est on X: %s' % str(X.shape)
        t0 = time()
        self.est.fit(X, y)
        print('Est fitted in %.2fm' % ((time() - t0) / 60.))
        self.clip_high = y.max()
        self.clip_low = y.min()
        return self

    def predict(self, X):
        n_days = X.shape[0]
        X = self.transform(X)

        pred = self.est.predict(X)

        pred = pred.reshape((n_days, self.n_stations, 4))
        pred = pred.mean(axis=2)
        if self.clip:
            pred = np.clip(pred, self.clip_low, self.clip_high)
        return pred


MODELS = {
    'baseline': Baseline,
    #'ensemble': EnsembledRegressor,
    #'dbn': DBNModel,
    'local': LocalModel,
    'kringing': KringingModel,
    'kriging': KringingModel,
    'ensemble_kriging': EnsembleKrigingModel,
    'perturbed_kriging': PertubedKrigingModel,
    'perturbed_pred_kriging': PertubedPredictionKrigingModel,
    'edgelocal': EdgeLocal,
    }
