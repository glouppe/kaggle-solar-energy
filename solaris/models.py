import numpy as np

from time import time
from scipy import ndimage
from collections import OrderedDict

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone


from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline

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

    ens_mean = True

    def __init__(self, est, intp_blocks=('nm_intp', 'nmft_intp', 'nm_intp_sigma'),
                 with_stationinfo=True, with_date=True,
                 with_solar=False, with_mask=False):
        self.est = est
        self.intp_blocks = intp_blocks
        self.with_date = with_date
        self.with_solar = with_solar
        self.with_mask = with_mask

    def fit(self, X_st, y):
        self.n_stations = y.shape[1]

        mask = None
        if self.with_mask:
            mask = util.clean_missing_labels(y)

        X = self.transform(X_st)
        y = self.transform_labels(y)

        if mask is not None:
            mask = self.transform_labels(mask)
            print('remove masked samples: %d' % mask.sum())
            y = y[~mask]
            X = X[~mask]

        self.est.fit(X, y)
        return self

    def predict(self, X_st):
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
            out.append(date)
            fx_names.append('doy')

        ## transform nm_intp to hourly features
        for b_name in self.intp_blocks:
            X = X_st[b_name]
            print b_name, X.shape
            # swap features and station interpolations
            X = np.swapaxes(X, 1, 3)
            fx_names.extend(['_'.join((b_name, fx, str(h)))
                             for fx in X_st.fx_name[b_name]
                             for h in range(X.shape[-2])])
            X = X.reshape((np.prod(X.shape[:2]), -1))
            out.append(X)

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
            X = X.reshape((np.prod(X.shape[:2]), -1))
            out.append(X)

        for b in out:
            print b.shape
        out = np.hstack(out)
        self.fx_names_ = fx_names
        print('transform to shape: %s' % str(out.shape))
        print('size of X in mb: %.2f' % (out.nbytes / 1024.0 / 1024.0))
        return out


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
        X_st['nm_intp'] = nm_intp
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


MODELS = {
    'baseline': Baseline,
    #'ensemble': EnsembledRegressor,
    #'dbn': DBNModel,
    #'local': LocalModel,
    'kringing': KringingModel,
    'kriging': KringingModel,
    'ensemble_kriging': EnsembleKrigingModel,
    'pertubed_kriging': PertubedKrigingModel,
    'pertubed_pred_kriging': PertubedPredictionKrigingModel,
    }
