import numpy as np
import cvxpy as cp
from itertools import combinations


def basis(K, T, P):
    Ps = np.atleast_1d(P)
    ws = [2 * np.pi / P for P in Ps]
    i_values = np.arange(1, K + 1)[:, np.newaxis]  # Column vector
    t_values = np.arange(T)  # Row vector
    # Computing the cos and sin matrices for each period
    B_cos_list = [np.cos(i_values * w * t_values).T for w in ws]
    B_sin_list = [np.sin(i_values * w * t_values).T for w in ws]

    # Interleave the results for each period using advanced indexing
    B_fourier = [np.empty((T, 2 * K), dtype=float) for _ in range(len(Ps))]
    for ix in range(len(Ps)):
        B_fourier[ix][:, ::2] = B_cos_list[ix]
        B_fourier[ix][:, 1::2] = B_sin_list[ix]

    # offset and linear terms
    v = np.sqrt(3)
    B_PL = np.linspace(-v, v, T).reshape(-1, 1)
    B_P0 = np.ones((T, 1))
    B0 = [B_PL, B_P0]

    # cross terms
    C = [cross_bases(*base_tuple) for base_tuple in combinations(B_fourier, 2)]

    B_list = B0 + B_fourier + C
    B = np.hstack(B_list)
    return B


def make_regularization_matrix(K, l, P):
    Ps = np.atleast_1d(P)
    ls_original = [l * (2 * np.pi) / np.sqrt(P) for P in Ps]
    ls_cross =
    l1 = l * (2 * np.pi) / np.sqrt(P1)
    l2 = l * (2 * np.pi) / np.sqrt(P2)
    l3 = l * (2 * np.pi) / np.sqrt(P3)
    l4 = l * (2 * np.pi) / np.sqrt(P2)
    l5 = l * (2 * np.pi) / np.sqrt(P3)
    l6 = l * (2 * np.pi) / np.sqrt(P3)

    # Create a sequence of values from 1 to K (repeated for cosine and sine)
    i_values = np.repeat(np.arange(1, K+1), 2)

    # Create blocks of coefficients
    block1 = i_values * l1
    block2 = i_values * l2
    block3 = i_values * l3
    block4 = np.tile(i_values * l4, 2 * K)
    block5 = np.tile(i_values * l5, 2 * K)
    block6 = np.tile(i_values * l6, 2 * K)

    # Combine the blocks to form the coefficient array
    coeff_i = np.concatenate(
        [[0, 0], block1, block2, block3, block4, block5, block6])

    # Create the diagonal matrix
    D = np.diag(coeff_i)

    return D



def individual_bases_3(K, T, P1, P2, P3):
    w1 = 2 * np.pi / P1
    w2 = 2 * np.pi / P2
    w3 = 2 * np.pi / P3

    i_values = np.arange(1, K+1)[:, np.newaxis]  # Column vector
    t_values = np.arange(T)  # Row vector

    # Computing the cos and sin matrices for each period
    B_P1_cos = np.cos(i_values * w1 * t_values).T
    B_P1_sin = np.sin(i_values * w1 * t_values).T

    B_P2_cos = np.cos(i_values * w2 * t_values).T
    B_P2_sin = np.sin(i_values * w2 * t_values).T

    B_P3_cos = np.cos(i_values * w3 * t_values).T
    B_P3_sin = np.sin(i_values * w3 * t_values).T

    # Interleave the results for each period using advanced indexing
    B_P1 = np.empty((T, 2*K), dtype=float)
    B_P1[:, ::2] = B_P1_cos
    B_P1[:, 1::2] = B_P1_sin

    B_P2 = np.empty((T, 2*K), dtype=float)
    B_P2[:, ::2] = B_P2_cos
    B_P2[:, 1::2] = B_P2_sin

    B_P3 = np.empty((T, 2*K), dtype=float)
    B_P3[:, ::2] = B_P3_cos
    B_P3[:, 1::2] = B_P3_sin

    # Add B_PL and B_P0
    v = np.sqrt(3)
    B_PL = np.linspace(-v, v, T).reshape(-1, 1)
    B_P0 = np.ones((T, 1))

    Base = [B_PL, B_P0, B_P1, B_P2, B_P3]
    return Base


def individual_bases_2(K, T, P1, P3):
    w1 = 2 * np.pi / P1
    w3 = 2 * np.pi / P3

    i_values = np.arange(1, K+1)[:, np.newaxis]  # Column vector
    t_values = np.arange(T)  # Row vector

    # Computing the cos and sin matrices for each period
    B_P1_cos = np.cos(i_values * w1 * t_values).T
    B_P1_sin = np.sin(i_values * w1 * t_values).T

    B_P3_cos = np.cos(i_values * w3 * t_values).T
    B_P3_sin = np.sin(i_values * w3 * t_values).T

    # Interleave the results for each period using advanced indexing
    B_P1 = np.empty((T, 2*K), dtype=float)
    B_P1[:, ::2] = B_P1_cos
    B_P1[:, 1::2] = B_P1_sin

    B_P3 = np.empty((T, 2*K), dtype=float)
    B_P3[:, ::2] = B_P3_cos
    B_P3[:, 1::2] = B_P3_sin

    # Add B_PL and B_P0
    v = np.sqrt(3)
    B_PL = np.linspace(-v, v, T).reshape(-1, 1)
    B_P0 = np.ones((T, 1))

    Base = [B_PL, B_P0, B_P1, B_P3]
    return Base


def cross_bases(B_P1, B_P2):
    # Reshape both arrays to introduce a new axis for broadcasting
    B_P1_new = B_P1[:, :, None]
    B_P2_new = B_P2[:, None, :]
    # Use broadcasting to compute the outer product for each row
    result = B_P1_new * B_P2_new
    # Reshape the result to the desired shape
    result = result.reshape(result.shape[0], -1)
    return result


def basis_3(K, T, P1, P2, P3):
    Bases = individual_bases_3(K, T, P1, P2, P3)
    B_PL = Bases[0]
    B_P0 = Bases[1]
    B_P1 = Bases[2]
    B_P2 = Bases[3]
    B_P3 = Bases[4]
    C_12 = cross_bases(B_P1, B_P2)
    C_13 = cross_bases(B_P1, B_P3)
    C_23 = cross_bases(B_P2, B_P3)
    B = np.hstack([B_PL, B_P0, B_P1, B_P2, B_P3, C_12, C_13, C_23])
    return B


def basis_2(K, T, P1, P3):
    Bases = individual_bases_2(K, T, P1, P3)
    B_PL = Bases[0]
    B_P0 = Bases[1]
    B_P1 = Bases[2]
    B_P3 = Bases[3]
    C_13 = cross_bases(B_P1, B_P3)
    B = np.hstack([B_PL, B_P0, B_P1, B_P3, C_13])
    return B


def regularization_matrix_3(K, l, P1, P2, P3):
    l1 = l * (2 * np.pi) / np.sqrt(P1)
    l2 = l * (2 * np.pi) / np.sqrt(P2)
    l3 = l * (2 * np.pi) / np.sqrt(P3)
    l4 = l * (2 * np.pi) / np.sqrt(P2)
    l5 = l * (2 * np.pi) / np.sqrt(P3)
    l6 = l * (2 * np.pi) / np.sqrt(P3)

    # Create a sequence of values from 1 to K (repeated for cosine and sine)
    i_values = np.repeat(np.arange(1, K+1), 2)

    # Create blocks of coefficients
    block1 = i_values * l1
    block2 = i_values * l2
    block3 = i_values * l3
    block4 = np.tile(i_values * l4, 2 * K)
    block5 = np.tile(i_values * l5, 2 * K)
    block6 = np.tile(i_values * l6, 2 * K)

    # Combine the blocks to form the coefficient array
    coeff_i = np.concatenate(
        [[0, 0], block1, block2, block3, block4, block5, block6])

    # Create the diagonal matrix
    D = np.diag(coeff_i)

    return D


def regularization_matrix_2(K, l, P1, P3):
    l1 = l * (2 * np.pi) / np.sqrt(P1)
    l3 = l * (2 * np.pi) / np.sqrt(P3)
    l5 = l * (2 * np.pi) / np.sqrt(P3)

    # Create a sequence of values from 1 to K (repeated for cosine and sine)
    i_values = np.repeat(np.arange(1, K+1), 2)

    # Create blocks of coefficients
    block1 = i_values * l1
    block2 = i_values * l3
    block3 = np.tile(i_values * l5, 2 * K)

    # Combine the blocks to form the coefficient array
    coeff_i = np.concatenate([[0, 0], block1, block2, block3])

    # Create the diagonal matrix
    D = np.diag(coeff_i)

    return D


def pinball_slopes(percentiles):
    percentiles = np.asarray(percentiles)
    a = (percentiles-50)*(0.01)
    b = (0.5)*np.ones((len(a),))
    return a, b


def fit_quantiles3(y1, K, P1, P2, P3, l, percentiles):
    T = len(y1)
    B = basis_3(K, T, P1, P2, P3)
    D = regularization_matrix_3(K, l, P1, P2, P3)
    a, b = pinball_slopes(percentiles)
    num_quantiles = len(a)
    Theta = cp.Variable((B.shape[1], num_quantiles))
    BT = B@Theta
    nonnanindex = ~np.isnan(y1)
    Var = y1[nonnanindex].reshape(-1, 1) - BT[nonnanindex]
    obj = cp.sum(Var@np.diag(a)+cp.abs(Var)@np.diag(b))
    # ensures quantiles are in order and prevents point masses ie minimum distance.
    cons = [cp.diff(BT, axis=1) >= 0.01]
    # cons+=[BT[:,0]>=0] #ensures quantiles are nonnegative
    Z = D @ Theta
    regularization = cp.sum_squares(Z)
    obj += regularization
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(verbose=True, solver=cp.MOSEK)
    Q = B@Theta.value
    return Q


def fit_quantiles2(y1, K, P1, P3, l, percentiles):
    T = len(y1)
    B = basis_2(K, T, P1, P3)
    D = regularization_matrix_2(K, l, P1, P3)
    a, b = pinball_slopes(percentiles)
    num_quantiles = len(a)
    Theta = cp.Variable((B.shape[1], num_quantiles))
    BT = B@Theta
    nonnanindex = ~np.isnan(y1)
    Var = y1[nonnanindex].reshape(-1, 1) - BT[nonnanindex]
    obj = cp.sum(Var@np.diag(a)+cp.abs(Var)@np.diag(b))
    # ensures quantiles are in order and prevents point masses ie minimum distance.
    cons = [cp.diff(BT, axis=1) >= 0.01]
    # cons+=[BT[:,0]>=0] #ensures quantiles are nonnegative
    Z = D @ Theta
    regularization = cp.sum_squares(Z)
    obj += regularization
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(verbose=True, solver=cp.MOSEK)
    Q = B@Theta.value
    return Q
