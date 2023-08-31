import numpy as np
from scipy.sparse import spdiags
from itertools import combinations


def make_basis_matrix(num_harmonics, length, periods, max_cross_k=None, custom_basis=None):
    if not (isinstance(custom_basis, dict) or custom_basis is None):
        raise TypeError("custom_basis should be a dictionary where the key is the index\n" +
                        "of the period and the value is a 2d array of appropriate size")
    num_harmonics = np.atleast_1d(num_harmonics)
    Ps = np.atleast_1d(periods)
    if len(num_harmonics) == 1:
        num_harmonics = np.tile(num_harmonics, len(Ps))
    elif len(num_harmonics) != len(Ps):
        raise ValueError("Please pass a single number of harmonics for all periods or a number for each period")

    ws = [2 * np.pi / P for P in Ps]
    i_value_list = [np.arange(1, nh + 1)[:, np.newaxis] for nh in num_harmonics]  # Column vector
    t_values = np.arange(length)  # Row vector
    # Computing the cos and sin matrices for each period
    B_cos_list = [np.cos(iv * w * t_values).T for w, iv in zip(ws, i_value_list)]
    B_sin_list = [np.sin(iv * w * t_values).T for w, iv in zip(ws, i_value_list)]

    # Interleave the results for each period using advanced indexing
    B_fourier = [np.empty((length, 2 * nh), dtype=float) for nh in num_harmonics]
    for ix in range(len(Ps)):
        B_fourier[ix][:, ::2] = B_cos_list[ix]
        B_fourier[ix][:, 1::2] = B_sin_list[ix]
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            B_fourier[ix] = val

    # offset and linear terms
    v = np.sqrt(3)
    B_PL = np.linspace(-v, v, length).reshape(-1, 1)
    B_P0 = np.ones((length, 1))
    B0 = [B_PL, B_P0]

    # cross terms, this handles the case of no cross terms gracefully (empty list)
    C = [cross_bases(*base_tuple, max_k=max_cross_k) for base_tuple in combinations(B_fourier, 2)]

    B_list = B0 + B_fourier + C
    B = np.hstack(B_list)
    return B


def make_regularization_matrix(num_harmonics, weight, periods, max_cross_k=None):
    num_harmonics = np.atleast_1d(num_harmonics)
    Ps = np.atleast_1d(periods)
    if len(num_harmonics) == 1:
        num_harmonics = np.tile(num_harmonics, len(Ps))
    elif len(num_harmonics) != len(Ps):
        raise ValueError("Please pass a single number of harmonics for all periods or a number for each period")
    ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]
    # this handles the case of no cross terms gracefully (empty list)
    ls_cross = [weight * (2 * np.pi) / np.sqrt(min(*c)) for c in combinations(Ps, 2)]
    cross_k_list = [c for c in combinations(num_harmonics, 2)]

    # Create a sequence of values from 1 to K (repeated for cosine and sine)
    i_value_list = [np.repeat(np.arange(1, nh + 1), 2) for nh in num_harmonics]
    if max_cross_k is None:
        max_cross_k = np.inf
    i_values_cross = [np.repeat(np.arange(1, min(max_cross_k, ck[1]) + 1), 2) for ck in cross_k_list]

    # Create blocks of coefficients
    blocks_original = [iv * lx for iv, lx in zip(i_value_list, ls_original)]
    blocks_cross = [np.tile(ivc * lx, 2 * min(max_cross_k, ck[0])) for ivc, lx, ck in
                    zip(i_values_cross, ls_cross, cross_k_list)]

    # Combine the blocks to form the coefficient array
    coeff_i = np.concatenate([np.zeros(2)] + blocks_original + blocks_cross)
    # Create the diagonal matrix
    D = spdiags(coeff_i, 0, coeff_i.size, coeff_i.size)

    return D


def cross_bases(B_P1, B_P2, max_k=None):
    if max_k is None:
        # Reshape both arrays to introduce a new axis for broadcasting
        B_P1_new = B_P1[:, :, None]
        B_P2_new = B_P2[:, None, :]
    else:
        B_P1_new = B_P1[:, :2*max_k, None]
        B_P2_new = B_P2[:, None, :2*max_k]
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
