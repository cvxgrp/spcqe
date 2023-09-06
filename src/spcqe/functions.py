import numpy as np
from scipy.sparse import spdiags
from itertools import combinations


def make_basis_matrix(num_harmonics, length, periods, max_cross_k=None, custom_basis=None):
    if not (isinstance(custom_basis, dict) or custom_basis is None):
        raise TypeError("custom_basis should be a dictionary where the key is the index\n" +
                        "of the period and the value is list containing the basis and the weights")
    num_harmonics = np.atleast_1d(num_harmonics)
    Ps = np.atleast_1d(periods)
    sort_idx = np.argsort(-Ps)
    Ps = -np.sort(-Ps)
    if len(num_harmonics) == 1:
        num_harmonics = np.tile(num_harmonics, len(Ps))
    elif len(num_harmonics) != len(Ps):
        raise ValueError("Please pass a single number of harmonics for all periods or a number for each period")
    # ensure if user has passed a list of harmonics, matching a list of periods, that we reorder that as well
    num_harmonics = num_harmonics[sort_idx]
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
            # also reorder index of custom basis, if necessary
            ixt = np.where(sort_idx == ix)[0][0]
            B_fourier[ixt] = val

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


def make_regularization_matrix(num_harmonics, weight, periods, max_cross_k=None, custom_basis=None):
    num_harmonics = np.atleast_1d(num_harmonics)
    Ps = np.atleast_1d(periods)
    sort_idx = np.argsort(-Ps)
    Ps = -np.sort(-Ps)
    if len(num_harmonics) == 1:
        num_harmonics = np.tile(num_harmonics, len(Ps))
    elif len(num_harmonics) != len(Ps):
        raise ValueError("Please pass a single number of harmonics for all periods or a number for each period")
    num_harmonics = num_harmonics[sort_idx]
    ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]
    # Create a sequence of values from 1 to K (repeated for cosine and sine)
    i_value_list = [np.repeat(np.arange(1, nh + 1), 2) for nh in num_harmonics]

    # Create blocks of coefficients
    blocks_original = [iv * lx for iv, lx in zip(i_value_list, ls_original)]
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            ixt = np.where(sort_idx == ix)[0][0]
            blocks_original[ixt] = ls_original[ixt] * np.arange(1, val.shape[1] + 1)
    if max_cross_k is not None:
        max_cross_k *= 2
    # this assumes  that the list of periods is ordered,  which is ensured in ln 51.  Ln 12 makes sure the bases are
    # in the same order
    blocks_cross = [[l2 for l1 in c[0][:max_cross_k] for l2 in c[1][:max_cross_k]]
                    for c in combinations(blocks_original, 2)]
    # This is *not* correct, as confirmed  with Stephen and Mehmet 8/31/23:
    # blocks_cross = [[max(l1, l2) for l1 in c[0][:max_cross_k] for l2 in c[1][:max_cross_k]] for c in
    #                 combinations(blocks_original, 2)]

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
