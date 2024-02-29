import numpy as np
from time import time
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import scipy.stats as stats
from spcqe.solvers import solve_cvx, solve_osd
from spcqe.functions import make_basis_matrix

QUANTILES = (0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.98)


class SmoothPeriodicQuantiles(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_harmonics,
        periods,
        standing_wave=False,
        trend=False,
        max_cross_k=None,
        quantiles=QUANTILES,
        weight=1,
        eps=0.01,
        standardize_data=True,
        take_log=False,
        solver="OSD",
        verbose=False,
        custom_basis=None,
    ):
        self.num_harmonics = num_harmonics
        self.periods = periods
        self.standing_wave = standing_wave
        self.trend = trend
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
        self.fit_parameters = None
        self.transform_parameters = None
        self.fit_time = None
        self._sc = None

    def fit(self, X, y=None):
        ti = time()
        data = np.asarray(X)
        if data.ndim != 1:
            raise AssertionError(
                "Data must be a scalar time series, castable as a 1d numpy array."
            )
        if self.take_log:
            new_data = np.nan * np.ones_like(data)
            msk = np.logical_and(~np.isnan(data), data >= 0)
            new_data[msk] = np.log(data[msk] + np.nanmin(data[data > 0]) * 1e-1)
            data = new_data
        if self.standardize_data:
            sc = StandardScaler()
            data = sc.fit_transform(data.reshape(-1, 1)).ravel()
            self._sc = sc
        else:
            # set to identity function
            self._sc = FunctionTransformer(lambda x: x)

        if self.solver.lower() in ["mosek", "osqp", "scs", "ecos", "clarabel"]:
            fit_quantiles, basis = solve_cvx(
                data,
                self.num_harmonics,
                self.periods,
                self.standing_wave,
                self.trend,
                self.max_cross_k,
                self.weight,
                self.quantiles,
                self.eps,
                self.solver.upper(),
                self.verbose,
                self.custom_basis,
            )
        elif self.solver.lower() in ["sig-decomp", "osd", "qss"]:
            fit_quantiles, basis = solve_osd(
                data,
                self.num_harmonics,
                self.periods,
                self.standing_wave,
                self.trend,
                self.max_cross_k,
                self.weight,
                self.quantiles,
                self.eps,
                self.solver.upper(),
                self.verbose,
                self.custom_basis,
            )
        else:
            raise NotImplementedError("non-cvxpy solution methods not yet implemented")
        self.basis = basis
        try:
            self.fit_quantiles = self._sc.inverse_transform(fit_quantiles)
        except ValueError:
            self.fit_quantiles = self._sc.inverse_transform(
                fit_quantiles.reshape(-1, 1)
            ).ravel()
        if self.take_log:
            self.fit_quantiles = np.exp(self.fit_quantiles)
        # refit basis weights after undoing any preprocessing
        self.fit_parameters, _, _, _ = np.linalg.lstsq(
            self.basis, self.fit_quantiles, rcond=None
        )
        self.length = len(data)
        tf = time()
        self.fit_time = tf - ti

    def predict(self, X, y=None):
        time_index = np.atleast_1d(X)
        newb = self.extend_basis(np.max(time_index) + 1)
        newb = newb[time_index]
        newq = newb @ self.fit_parameters
        newq = np.sort(newq, axis=1)
        return newq

    def transform(self, X, y=None):
        # Currently this function calculates the transforms on the fly each time the method is called. This is not
        # unreasonable, as fitting the quantile basis parameters (with convex optimization) is the truly time intensive
        # part. However, this function could be further optimized with memoization---if we are asked to transform data
        # during a time period we've seen before, we could look that up rather than recalculating the linear
        # coefficients for the transform. That said, this function only takes a handful of microseconds to execute on
        # >300,000 data points, and it is O(T), that is, linear computational complexity with respect to the size of the
        # time axis. So, this further optimization is probably not a huge priority.
        data = np.asarray(X)
        if len(data) != self.length and y is None:
            raise ValueError(
                "If not transforming the original fit data set, a time index must be passed as y"
            )
        # get correct basis matrix and quantile estimates for time period of prediction
        if y is not None:
            new_quantiles = self.predict(y)
        else:
            new_quantiles = self.fit_quantiles
        # fit piecewise linear transforms, one for each time index
        # start by making a piecewise linear basis matrix with known knot points for each time index: T x q x q
        mats = np.empty(
            (new_quantiles.shape[0], len(self.quantiles), len(self.quantiles))
        )
        for jx in range(len(self.quantiles)):
            if jx == 0:
                mats[:, :, jx] = 1
            elif jx == 1:
                mats[:, :, jx] = new_quantiles
            else:
                mats[:, :, jx] = np.clip(
                    new_quantiles
                    - np.tile(new_quantiles[:, jx - 1], (new_quantiles.shape[1], 1)).T,
                    0,
                    np.inf,
                )
        # This works but has a loop over the time index which is >> than the number of quantiles
        # for ix in range(new_quantiles.shape[0]):
        #     mats[ix] = self.x_expand(new_quantiles[ix], ix)
        # the LHS of each matrix equation is the same and does not change over time
        yy = stats.norm.ppf(self.quantiles)[np.newaxis, :]
        # solve vectorized matrix equations, T independent (q x q) set of equations
        parameters = np.linalg.solve(mats, yy)
        # apply the transform to the new data: this makes the PWL basis expansion for the new data (T x q)
        mats = np.empty((new_quantiles.shape[0], len(self.quantiles)))
        for jx in range(len(self.quantiles)):
            if jx == 0:
                mats[:, jx] = 1
            elif jx == 1:
                mats[:, jx] = data
            else:
                mats[:, jx] = np.clip(data - new_quantiles[:, jx - 1], 0, np.inf)
        # Finally, apply all the transforms, one for each time index, i
        Z = np.einsum("ij, ij -> i", mats, parameters)
        return Z

    def inverse_transform(self, X, y=None):
        data = np.asarray(X)
        if len(data) != self.length and y is None:
            raise ValueError(
                "If not transforming the original fit data set, a time index must be passed as y"
            )
        # get correct basis matrix and quantile estimates for time period of prediction
        if y is not None:
            new_quantiles = self.predict(y)
        else:
            new_quantiles = self.fit_quantiles
        # fit piecewise linear transforms, one for each time index
        # make the piecewise linear basis matrix with known knot points, for the inverse this is static in time: (q x q)
        mats = np.empty((len(self.quantiles), len(self.quantiles)))
        for jx in range(len(self.quantiles)):
            if jx == 0:
                mats[:, jx] = 1
            elif jx == 1:
                mats[:, jx] = stats.norm.ppf(self.quantiles)
            else:
                mats[:, jx] = np.clip(
                    stats.norm.ppf(self.quantiles)
                    - stats.norm.ppf(self.quantiles)[jx - 1],
                    0,
                    np.inf,
                )
        # add a new axis for vectorized numpy matrix solve
        mats = mats[np.newaxis, :, :]
        # in the inverse, there is now a different LHS for each matrix equation
        yy = new_quantiles
        parameters = np.linalg.solve(mats, yy)  # shape len(data) x len(quantiles)
        # apply the transform to the new data (T x q)
        mats = np.empty((new_quantiles.shape[0], len(self.quantiles)))
        for jx in range(len(self.quantiles)):
            if jx == 0:
                mats[:, jx] = 1
            elif jx == 1:
                mats[:, jx] = data
            else:
                mats[:, jx] = np.clip(
                    data - stats.norm.ppf(self.quantiles)[jx - 1], 0, np.inf
                )
        Z = np.einsum("ij, ij -> i", mats, parameters)
        return Z

    def x_expand(self, xi, tix):
        xin = np.atleast_1d(xi)
        h1 = np.ones_like(xin)
        h2 = xin
        h3up = [np.clip(xin - kn, 0, np.inf) for kn in self.fit_quantiles[tix][1:-1]]
        basis = np.r_[[h1, h2] + h3up].T
        return basis

    def score(self, X, y=None):
        data = np.asarray(X)
        if len(data) != self.length and y is None:
            raise ValueError(
                "If not transforming the original fit data set, a time index must be passed as y"
            )
        # get correct basis matrix and quantile estimates for time period of prediction
        if y is not None:
            new_quantiles = self.predict(y)
        else:
            new_quantiles = self.fit_quantiles
        data = data[:, np.newaxis]
        q = self.quantiles[np.newaxis, :]
        score = np.sum(
            np.trapz(
                0.5 * np.abs(data - new_quantiles) + (q - 0.5) * (data - new_quantiles),
                x=self.quantiles,
            )
        )
        return score

    def extend_basis(self, t):
        T = self.basis.shape[0]
        if 0 <= t < T:
            return self.basis
        elif t >= T:
            new_basis = make_basis_matrix(
                self.num_harmonics,
                t,
                self.periods,
                max_cross_k=self.max_cross_k,
                custom_basis=self.custom_basis,
            )
            return new_basis
        else:
            raise NotImplementedError(
                "Extending the basis to time before the training data not currently supported"
            )
