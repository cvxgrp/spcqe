import numpy as np
from time import time
from sklearn.base import TransformerMixin, BaseEstimator
from spcqe.solvers import solve_cvx

QUANTILES = (.02, .10, .20, .30, .40, .50, .60, .70, .80, .90, .98)


class SmoothPeriodicQuantiles(BaseEstimator, TransformerMixin):
    def __init__(self, num_harmonics, periods, quantiles=QUANTILES, weight=1, eps=0.01, solver='MOSEK',
                 verbose=False):
        self.num_harmonics = num_harmonics
        self.periods = periods
        self.quantiles = np.atleast_1d(np.asarray(quantiles))
        self.weight = weight
        self.eps = eps
        self.solver = solver
        self.verbose = verbose
        self.length = None
        self.basis = None
        self.fit_quantiles = None
        self.fit_time = None

    def fit(self, X, y=None):
        ti = time()
        data = np.asarray(X)
        if data.ndim != 1:
            raise AssertionError("Data must be a scalar time series, castable as a 1d numpy array.")
        self.length = len(data)
        if self.solver.lower() in ['mosek', 'osqp', 'scs', 'ecos']:
            fit_quantiles, basis = solve_cvx(data, self.num_harmonics, self.periods, self.weight, self.quantiles,
                                         self.eps, solver=self.solver.upper(), verbose=self.verbose)
        else:
            raise NotImplementedError('non-cvxpy solution methods not yet implemented')
        self.basis = basis
        self.fit_quantiles = fit_quantiles
        tf = time()
        self.fit_time = tf - ti
