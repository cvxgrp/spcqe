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

    # cross terms, this handles the case of no cross terms gracefully (empty list)
    C = [cross_bases(*base_tuple) for base_tuple in combinations(B_fourier, 2)]

    B_list = B0 + B_fourier + C
    B = np.hstack(B_list)
    return B


def make_regularization_matrix(K, l, P):
    Ps = np.atleast_1d(P)
    ls_original = [l * (2 * np.pi) / np.sqrt(P) for P in Ps]
    # this handles the case of no cross terms gracefully (empty list)
    ls_cross = [l * (2 * np.pi) / np.sqrt(max(*c)) for c in combinations(Ps, 2)]

    # Create a sequence of values from 1 to K (repeated for cosine and sine)
    i_values = np.repeat(np.arange(1, K+1), 2)

    # Create blocks of coefficients
    blocks_original = [i_values * lx for lx in ls_original]
    blocks_cross = [np.tile(i_values * lx, 2*K) for lx in ls_cross]

    # Combine the blocks to form the coefficient array
    coeff_i = np.concatenate([np.zeros(2)] + blocks_original + blocks_cross)

    # Create the diagonal matrix
    D = np.diag(coeff_i)

    return D


def cross_bases(B_P1, B_P2):
    # Reshape both arrays to introduce a new axis for broadcasting
    B_P1_new = B_P1[:, :, None]
    B_P2_new = B_P2[:, None, :]
    # Use broadcasting to compute the outer product for each row
    result = B_P1_new * B_P2_new
    # Reshape the result to the desired shape
    result = result.reshape(result.shape[0], -1)
    return result