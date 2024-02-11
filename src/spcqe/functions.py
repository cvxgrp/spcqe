import numpy as np
from scipy.sparse import spdiags
from itertools import combinations


def make_basis_matrix(num_harmonics, length, periods, standing_wave=False, trend=False, max_cross_k=None, custom_basis=None):

    # Sanity checks
    if not (isinstance(custom_basis, dict) or custom_basis is None):
        raise TypeError("custom_basis should be a dictionary where the key is the index\n" +
                        "of the period and the value is list containing the basis and the weights")
    Ps = np.atleast_1d(periods)
    num_harmonics = np.atleast_1d(num_harmonics)
    if len(num_harmonics) == 1:
        num_harmonics = np.tile(num_harmonics, len(Ps))
    elif len(num_harmonics) != len(Ps):
        raise ValueError("Please pass a single number of harmonics for all periods or a number for each period")
    standing_wave = np.atleast_1d(standing_wave)
    if len(standing_wave) == 1:
        standing_wave = np.tile(standing_wave, len(Ps))
    elif len(standing_wave) != len(Ps):
        raise ValueError("Please pass a single boolean for standing_wave for all periods or a boolean for each period")
    
    # Sort the periods and harmonics
    sort_idx = np.argsort(-Ps)
    Ps = -np.sort(-Ps) # Sort in descending order
    num_harmonics = num_harmonics[sort_idx]
    standing_wave = standing_wave[sort_idx]

    # Make the basis
    t_values = np.arange(length) # Time stamps (row vector)
    B_fourier = []
    for ix, P in enumerate(Ps):
        i_values = np.arange(1, num_harmonics[ix] + 1)[:, np.newaxis] # Harmonic indices (column vector)
        if standing_wave[ix]:
            w = 2 * np.pi / (P*2)
            B_sin = np.sin(i_values * w * np.mod(t_values, P))
            B_f = np.empty((length, num_harmonics[ix]), dtype=float)
            B_f[:] = B_sin.T
        else:
            w = 2 * np.pi / P
            B_cos = np.cos(i_values * w * t_values)
            B_sin = np.sin(i_values * w * t_values)
            B_f = np.empty((length, 2 * num_harmonics[ix]), dtype=float)
            B_f[:, ::2] = B_cos.T
            B_f[:, 1::2] = B_sin.T
        B_fourier.append(B_f)
    
    # Use custom basis if provided
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            # check length
            if val.shape[0] != length:
                # extend to cover future time period if necessary
                multiplier = max(1, val.shape[0] // length + 1)
                new_val = np.tile(val, (multiplier, 1))[:length]
            else:
                new_val = val[:length]
            # also reorder index of custom basis, if necessary
            ixt = np.where(sort_idx == ix)[0][0]
            B_fourier[ixt] = new_val

    # Add offset and linear terms
    if trend is False:
        B_P0 = np.ones((length, 1))
        B0 = [B_P0]
    else:
        v = np.sqrt(3)
        B_PL = np.linspace(-v, v, length).reshape(-1, 1)
        B_P0 = np.ones((length, 1))
        B0 = [B_PL, B_P0]

    # Cross terms, this handles the case of no cross terms gracefully (empty list)
    C = [cross_bases(*base_tuple, max_k=max_cross_k) for base_tuple in combinations(B_fourier, 2)]

    B_list = B0 + B_fourier + C
    B = np.hstack(B_list)
    return B

# TODO: is it different if standing wave is True?
def make_regularization_matrix(num_harmonics, weight, periods, standing_wave=False, trend=False, max_cross_k=None, custom_basis=None):
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
    if trend is False:
        first_block = [np.zeros(1)]
    else:
        first_block = [np.zeros(2)]
    coeff_i = np.concatenate(first_block + blocks_original + blocks_cross)
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
