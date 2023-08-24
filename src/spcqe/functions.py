import numpy as np
import cvxpy as cp
from itertools import combinations


def basis(num_harmonics, length, periods):
    Ps = np.atleast_1d(periods)
    ws = [2 * np.pi / P for P in Ps]
    i_values = np.arange(1, num_harmonics + 1)[:, np.newaxis]  # Column vector
    t_values = np.arange(length)  # Row vector
    # Computing the cos and sin matrices for each period
    B_cos_list = [np.cos(i_values * w * t_values).T for w in ws]
    B_sin_list = [np.sin(i_values * w * t_values).T for w in ws]

    # Interleave the results for each period using advanced indexing
    B_fourier = [np.empty((length, 2 * num_harmonics), dtype=float) for _ in range(len(Ps))]
    for ix in range(len(Ps)):
        B_fourier[ix][:, ::2] = B_cos_list[ix]
        B_fourier[ix][:, 1::2] = B_sin_list[ix]

    # offset and linear terms
    v = np.sqrt(3)
    B_PL = np.linspace(-v, v, length).reshape(-1, 1)
    B_P0 = np.ones((length, 1))
    B0 = [B_PL, B_P0]

    # cross terms, this handles the case of no cross terms gracefully (empty list)
    C = [cross_bases(*base_tuple) for base_tuple in combinations(B_fourier, 2)]

    B_list = B0 + B_fourier + C
    B = np.hstack(B_list)
    return B


def make_regularization_matrix(num_harmonics, weight, periods):
    Ps = np.atleast_1d(periods)
    ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]
    # this handles the case of no cross terms gracefully (empty list)
    ls_cross = [weight * (2 * np.pi) / np.sqrt(min(*c)) for c in combinations(Ps, 2)]

    # Create a sequence of values from 1 to K (repeated for cosine and sine)
    i_values = np.repeat(np.arange(1, num_harmonics + 1), 2)

    # Create blocks of coefficients
    blocks_original = [i_values * lx for lx in ls_original]
    blocks_cross = [np.tile(i_values * lx, 2 * num_harmonics) for lx in ls_cross]

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


def pinball_slopes(quantiles):
    percentiles = np.asarray(quantiles)
    a = (quantiles-.5)
    b = (0.5)*np.ones((len(a),))
    return a, b
