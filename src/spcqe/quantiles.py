import numpy as np
from time import time
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from spcqe.solvers import solve_cvx, solve_osd

QUANTILES = (.02, .10, .20, .30, .40, .50, .60, .70, .80, .90, .98)


class SmoothPeriodicQuantiles(BaseEstimator, TransformerMixin):
    def __init__(self, num_harmonics, periods, max_cross_k=None, quantiles=QUANTILES, weight=1, eps=0.01, standardize_data=True,
                 take_log=False, solver='OSD', verbose=False, custom_basis=None):
        self.num_harmonics = num_harmonics
        self.periods = periods
        self.max_cross_k = max_cross_k
        self.quantiles = np.atleast_1d(np.asarray(quantiles))
        self.weight = weight
        self.eps = eps
        self.solver = solver
        self.standardize_data = standardize_data
        self.take_log = take_log
        self.verbose = verbose
        self.custom_basis = custom_basis
        self.length = None
        self.basis = None
        self.fit_quantiles = None
        self.fit_time = None
        self._sc = None

    def fit(self, X, y=None):
        ti = time()
        data = np.asarray(X)
        if data.ndim != 1:
            raise AssertionError("Data must be a scalar time series, castable as a 1d numpy array.")
        if self.take_log:
            new_data = np.nan * np.ones_like(data)
            msk = np.logical_and(~np.isnan(data), data >= 0)
            new_data[msk] = np.log(data[msk] + np.nanmin(data[data > 0]) * 1e-1)
            data = new_data
        if self.standardize_data:
            sc = StandardScaler()
            data = sc.fit_transform(data.reshape(-1,1)).ravel()
            self._sc = sc
        else:
            # set to identity function
            self._sc = FunctionTransformer(lambda x: x)
        self.length = len(data)
        if self.solver.lower() in ['mosek', 'osqp', 'scs', 'ecos', 'clarabel']:
            fit_quantiles, basis = solve_cvx(data, self.num_harmonics, self.periods, self.max_cross_k, self.weight,
                                             self.quantiles, self.eps, self.solver.upper(), self.verbose,
                                             self.custom_basis)
        elif self.solver.lower() in ['sig-decomp', 'osd', 'qss']:
            fit_quantiles, basis = solve_osd(data, self.num_harmonics, self.periods, self.max_cross_k, self.weight,
                                             self.quantiles, self.eps, self.solver.upper(), self.verbose,
                                             self.custom_basis)
        else:
            raise NotImplementedError('non-cvxpy solution methods not yet implemented')
        self.basis = basis
        try:
            self.fit_quantiles = self._sc.inverse_transform(fit_quantiles)
        except ValueError:
            self.fit_quantiles = self._sc.inverse_transform(fit_quantiles.reshape(-1, 1)).ravel()
        if self.take_log:
            self.fit_quantiles = np.exp(self.fit_quantiles)
        tf = time()
        self.fit_time = tf - ti
