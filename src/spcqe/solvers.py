import cvxpy as cp
import numpy as np
from spcqe.functions import basis, make_regularization_matrix, pinball_slopes


def solve_cvx(data, num_harmonics, periods, weight, quantiles, eps, solver, verbose):
    problem, basis = make_cvx_problem(data, num_harmonics, periods, weight, quantiles, eps)
    problem.solve(solver=solver, verbose=verbose)
    theta = problem.variables()[0]
    quantiles = basis @ theta.value
    return quantiles, basis


def make_cvx_problem(data, num_harmonics, periods, weight, quantiles, eps):
    length = len(data)
    B = basis(num_harmonics, length, periods)
    D = make_regularization_matrix(num_harmonics, weight, periods)
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
