import cvxpy as cp
import numpy as np
from spcqe.functions import (
    make_basis_matrix,
    make_regularization_matrix,
    pinball_slopes,
)
from gfosd import Problem as ProblemOSD
from gfosd.components import Basis, SumQuantile
from tqdm import tqdm


def solve_cvx(
    data,
    num_harmonics,
    periods,
    standing_wave,
    trend,
    max_cross_k,
    weight,
    quantiles,
    eps,
    solver,
    verbose,
    custom_basis,
):
    if len(quantiles) > 1:
        problem, basis = make_cvx_problem(
            data,
            num_harmonics,
            periods,
            standing_wave,
            trend,
            max_cross_k,
            weight,
            quantiles,
            eps,
            custom_basis,
        )
    else:
        problem, basis = make_cvx_problem_single(
            data,
            num_harmonics,
            periods,
            standing_wave,
            trend,
            max_cross_k,
            weight,
            quantiles,
            custom_basis,
        )
    if solver.lower() == "clarabel":
        problem.solve(
            solver=solver,
            verbose=verbose,
            tol_gap_abs=1e-3,
            tol_gap_rel=1e-3,
            tol_feas=1e-3,
            tol_infeas_abs=1e-3,
            tol_infeas_rel=1e-3,
        )
    else:
        problem.solve(solver=solver, verbose=verbose)
    theta = problem.variables()[0]
    quantile_estimates = basis @ theta.value
    return quantile_estimates, basis


def solve_osd(
    data,
    num_harmonics,
    periods,
    standing_wave,
    trend,
    max_cross_k,
    weight,
    quantiles,
    eps,
    solver,
    verbose,
    custom_basis,
):
    length = len(data)
    basis = make_basis_matrix(
        num_harmonics,
        length,
        periods,
        standing_wave,
        trend,
        max_cross_k=max_cross_k,
        custom_basis=custom_basis,
    )
    reg = make_regularization_matrix(
        num_harmonics,
        weight,
        periods,
        standing_wave,
        trend,
        max_cross_k=max_cross_k,
        custom_basis=custom_basis,
    )
    if len(quantiles) > 1:
        problems = [
            make_osd_problem(data, num_harmonics, periods, weight, q, basis, reg)
            for q in quantiles
        ]
        quantile_estimates = np.zeros((len(data), len(quantiles)), dtype=float)
        for ix, problem in tqdm(enumerate(problems), total=len(quantiles), ncols=80):
            if np.abs(quantiles[ix] - 0.5) > 0.4:
                osd_abs = 1e-4
            else:
                osd_abs = 1e-3
            if solver == 'qss':
                problem.decompose(
                    verbose=verbose,
                    solver=solver,
                    rho_update="none",
                    rho=[0.5, 0.02],
                    max_iter=5000,
                    eps_abs=osd_abs,
                    eps_rel=osd_abs,
                )
            else:
                problem.decompose(
                    verbose=verbose,
                    solver=solver,
                    eps_abs=osd_abs,
                    eps_rel=osd_abs,
                )
            quantile_estimates[:, ix] = problem.decomposition[1]
        quantile_estimates = np.sort(quantile_estimates, axis=1)
    else:
        problem = make_osd_problem(
            data, num_harmonics, periods, weight, quantiles, basis, reg
        )
        # problem.decompose(verbose=verbose, rho_update="none", rho=[10, .1], max_iter=5000, eps_abs=1e-3, eps_rel=1e-3)
        problem.decompose(
            verbose=verbose,
            solver=solver,
            rho_update="none",
            rho=[0.5, 0.02],
            max_iter=5000,
            eps_abs=1e-3,
            eps_rel=1e-3,
        )
        # problem.decompose(verbose=verbose, max_iter=5000, eps_abs=1e-3, eps_rel=1e-3)
        quantile_estimates = problem.decomposition[1]
    return quantile_estimates, basis


def make_cvx_problem(
    data,
    num_harmonics,
    periods,
    standing_wave,
    trend,
    max_cross_k,
    weight,
    quantiles,
    eps,
    custom_basis,
):
    length = len(data)
    B = make_basis_matrix(
        num_harmonics,
        length,
        periods,
        standing_wave,
        trend,
        max_cross_k=max_cross_k,
        custom_basis=custom_basis,
    )
    D = make_regularization_matrix(
        num_harmonics,
        weight,
        periods,
        standing_wave,
        trend,
        max_cross_k=max_cross_k,
        custom_basis=custom_basis,
    )
    num_quantiles = len(quantiles)
    a, b = pinball_slopes(quantiles)
    Theta = cp.Variable((B.shape[1], num_quantiles))
    BT = B @ Theta
    nonnanindex = ~np.isnan(data)
    Var = data[nonnanindex].reshape(-1, 1) - BT[nonnanindex]
    obj = cp.sum(Var @ np.diag(a) + cp.abs(Var) @ np.diag(b))
    # ensures quantiles are in order and prevents point masses ie minimum distance.
    if num_quantiles > 1:
        cons = [cp.diff(BT, axis=1) >= eps]
    else:
        cons = []
    # cons+=[BT[:,0]>=0] #ensures quantiles are nonnegative
    Z = D @ Theta
    regularization = cp.sum_squares(Z)
    obj += regularization
    prob = cp.Problem(cp.Minimize(obj), cons)
    return prob, B


def make_cvx_problem_single(
    data,
    num_harmonics,
    periods,
    standing_wave,
    trend,
    max_cross_k,
    weight,
    quantile,
    custom_basis,
):
    length = len(data)
    B = make_basis_matrix(
        num_harmonics,
        length,
        periods,
        standing_wave,
        trend,
        max_cross_k=max_cross_k,
        custom_basis=custom_basis,
    )
    D = make_regularization_matrix(
        num_harmonics,
        weight,
        periods,
        standing_wave,
        trend,
        max_cross_k=max_cross_k,
        custom_basis=custom_basis,
    )
    theta = cp.Variable(B.shape[1])
    q_hat = B @ theta
    pinball_loss = lambda x: cp.sum(0.5 * cp.abs(x) + (quantile[0] - 0.5) * x)
    loss = pinball_loss(q_hat - data)
    reg = cp.sum_squares(D @ theta)
    prob = cp.Problem(cp.Minimize(loss + reg))
    return prob, B


def make_osd_problem(
    data, num_harmonics, periods, weight, quantile, basis_matrix, reg_matrix
):
    B = basis_matrix
    D = reg_matrix
    x1 = SumQuantile(tau=quantile)
    x2 = Basis(basis=B, penalty=D)
    prob = ProblemOSD(data, [x1, x2])
    return prob
