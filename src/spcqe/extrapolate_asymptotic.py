import warnings

import numpy as np
import scipy.stats as sps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

dist = sps.norm()

def get_asymptote_parameters_out(x0, _y0, x1, _y1, yasympt):
    y0, y1 = dist.ppf(_y0), dist.ppf(_y1)
    yprime0 = (y0 - y1) / (x0 - x1)
    beta = yprime0 / (y0 - yasympt)

    exp_input = -beta * x0
    alpha = np.where(
        np.isinf(exp_input) | np.isnan(exp_input),
        np.inf,  # Return +inf for invalid exp inputs
        - (yasympt - y0) * np.exp(exp_input)
    )
    return alpha, beta

def get_asymptote_parameters_in(x0, _y0, x1, _y1, xasympt):
    y0, y1 = dist.ppf(_y0), dist.ppf(_y1)
    yprime0 = (y1 - y0) / (x1 - x0)
    beta = yprime0 * (x0 - xasympt)

    # Create a mask for valid x0 values
    valid_mask = x0 > xasympt
    exp_input = np.where(valid_mask, y0 / beta, np.inf)  # Default to +inf if invalid

    alpha = np.where(
        np.isinf(exp_input) | np.isnan(exp_input),
        np.inf,  # Return +inf for invalid exp inputs
        1 / (x0 - xasympt) * np.exp(exp_input)
    )
    return alpha, beta

def asymptote_out(x, yasympt, alpha, beta):
    exp_input = beta * x
    return np.where(
        np.isinf(exp_input) | np.isnan(exp_input),
        np.inf,  # Return +inf for invalid exp inputs
        yasympt + alpha * np.exp(exp_input)
    )

def asymptote_in(x, xasympt, alpha, beta):
    log_input = alpha * (x - xasympt)
    return np.where(
        (x <= xasympt) | (log_input <= 0),
        -np.inf,  # Return -inf for invalid conditions
        beta * np.log(log_input)
    )

def inverse_asymptote_out(y, yasympt, alpha, beta):
    log_input = (y - yasympt) / alpha
    return np.where(
        log_input <= 0,
        -np.inf,  # Return -inf for invalid log input
        np.log(log_input) / beta
    )

def inverse_asymptote_in(y, xasympt, alpha, beta):
    exp_input = y / beta
    return np.where(
        np.isinf(exp_input) | np.isnan(exp_input),
        np.inf,  # Return +inf for invalid exp inputs
        xasympt + np.exp(exp_input) / alpha
    )


def plot_pdf(ax, transf, label):
    x = np.linspace(-4, 4, 1000)
    y = sps.norm.pdf(x)
    sns.histplot(transf, bins=1000, stat='density', ax=ax, kde=False, label=label)
    ax.plot(x, y, 'r', label='Normal PDF')
    ax.set_xlabel('Transformed Signal')
    ax.legend()
    return ax

def find_idx_in_signal(tail, idx_in_tail, transf):
    return np.arange(len(transf))[tail][idx_in_tail]

def plot_tails(ax, sig, quantiles, fit_quantiles, transf, params, key, asymptote, space, index, extrap_width):
               
    #           space, asymp, idx_tail, extrap_xs):
    # params = MyAsympTransformer.parameters[tail]
    
    linear_interp = interp1d(fit_quantiles[index], dist.ppf(quantiles), kind='linear', fill_value='extrapolate')
    if key == 'upper':
        sig_values = np.linspace(fit_quantiles[index, 0], sig[index], 100)
        extrap_values = np.linspace(fit_quantiles[index, -1], fit_quantiles[index, -1] + extrap_width, 100)
        if space == 'output':
            extrap_ys = asymptote_out(extrap_values, asymptote, params[0][index], params[1][index])
        elif space == 'input':
            extrap_ys = asymptote_in(extrap_values, asymptote, params[0][index], params[1][index])
    elif key == 'lower':
        sig_values = np.linspace(sig[index], fit_quantiles[index, -1], 100)
        extrap_values = np.linspace(fit_quantiles[index, 0] - extrap_width, fit_quantiles[index, 0], 100)
        if space == 'output':
            extrap_ys = asymptote_out(extrap_values, asymptote, params[0][index], params[1][index])
        elif space == 'input':
            extrap_ys = asymptote_in(extrap_values, asymptote, params[0][index], params[1][index])
    ax.scatter(fit_quantiles[index], dist.ppf(quantiles), color='C3', marker='+', label='quantiles setpoints')
    ax.axvline(sig[index], color='C3', linestyle='--', label='value to transform')
    ax.plot(sig_values, linear_interp(sig_values), label='linear transform function')
    #ax.scatter(sig[index], transf[index], marker='x', label='transformed value - linear')
    ax.plot(
        extrap_values,
        extrap_ys,
        linestyle='--',
        color='black',
        label='asymptotic outer transform function'
        )
    ax.scatter(sig[index], transf[index], color='black', marker='x', label='transformed value')
    ax.set_title(f'Transfer function from dilated signal to normal distribution - {key} tail')
    ax.legend()
    return ax