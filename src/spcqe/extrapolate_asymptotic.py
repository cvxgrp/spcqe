import warnings

import numpy as np
import scipy.stats as sps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

dist = sps.norm()
XSOLAR = -0.025
YSOLAR = dist.ppf(0.99999)

def get_asymptote_parameters_out(x0, _y0, x1, _y1, yasympt):
    # Looking for a function of type yasympt + alpha * exp(beta * x)
    # With C1 continuity constraints at (x0, y0)
    y0, y1 = dist.ppf(_y0), dist.ppf(_y1)
    yprime0 = (y0 - y1) / (x0 - x1)
    beta = yprime0 / (y0 - yasympt)
    exp_input = -beta * x0
    exp_output = safe_exp(exp_input)
    alpha = - (yasympt - y0) * exp_output
    return alpha, beta

def get_asymptote_parameters_in(x0, _y0, x1, _y1, xasympt):
    # Looking for a function of type beta * np.log(alpha *(x - xasympt))
    # With C1 continuity constraints at (x0, y0)
    y0, y1 = dist.ppf(_y0), dist.ppf(_y1)
    yprime0 = (y1 - y0) / (x1 - x0)
    beta = yprime0 * (x0 - xasympt)
    exp_input = y0 / beta
    exp_output = safe_exp(exp_input)
    alpha = 1 / (x0 - xasympt) * exp_output
    return alpha, beta

def asymptote_out(x, yasympt, alpha, beta):
    exp_input = beta * x
    exp_output = safe_exp(exp_input)
    return yasympt + alpha * exp_output


def asymptote_in(x, xasympt, alpha, beta):
    log_input = alpha * (x - xasympt)
    log_output = safe_log(log_input)
    return beta * log_output

def inverse_asymptote_out(y, yasympt, alpha, beta):
    log_input = (y - yasympt) / alpha
    log_output = safe_log(log_input)
    return log_output / beta

def inverse_asymptote_in(y, xasympt, alpha, beta):
    exp_input = y / beta
    exp_output = safe_exp(exp_input)
    return xasympt + exp_output / alpha

def safe_exp(exp_input):
    exp_output = np.empty_like(exp_input)
    exp_output[exp_input > 709] = np.inf
    exp_output[exp_input < -709] = 0
    exp_output[(-709 <= exp_input) & (exp_input <= 709)] = np.exp(exp_input[(-709 <= exp_input) & (exp_input <= 709)])
    return exp_output

def safe_log(log_input):
    log_output = np.empty_like(log_input)
    log_output[log_input <= 0] = -np.inf
    log_output[log_input > 0] = np.log(log_input[log_input > 0])
    return log_output

def init_extrap(extrapolate):
    if extrapolate == 'linear':
        return {'lower': 'linear', 'upper': 'linear'}
    elif extrapolate == 'solar':
        extrapolate = {'lower':(XSOLAR, 'input'), 'upper':(YSOLAR, 'output')}
    return extrapolate


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

def plot_tails(ax, sig, quantiles, fit_quantiles, transf, method, key, index, extrap_width,
               params=None,
               asymptote=None,
               space=None):
    
    linear_interp = interp1d(fit_quantiles[index], dist.ppf(quantiles), kind='linear', fill_value='extrapolate')
    if key == 'upper':
        sig_values = np.linspace(fit_quantiles[index, 0], sig[index], 100)
        extrap_values = np.linspace(fit_quantiles[index, -1], fit_quantiles[index, -1] + extrap_width, 100)
        if method == 'asymptotic':
            if space == 'output':
                extrap_ys = asymptote_out(extrap_values, asymptote, params[0][index], params[1][index])
            elif space == 'input':
                extrap_ys = asymptote_in(extrap_values, asymptote, params[0][index], params[1][index])
        elif method == 'linear':
            extrap_ys = linear_interp(extrap_values)
    elif key == 'lower':
        sig_values = np.linspace(sig[index], fit_quantiles[index, -1], 100)
        extrap_values = np.linspace(fit_quantiles[index, 0] - extrap_width, fit_quantiles[index, 0], 100)
        if method == 'asymptotic':
            if space == 'output':
                extrap_ys = asymptote_out(extrap_values, asymptote, params[0][index], params[1][index])
            elif space == 'input':
                extrap_ys = asymptote_in(extrap_values, asymptote, params[0][index], params[1][index])
        elif method == 'linear':
            extrap_ys = linear_interp(extrap_values)
    ax.scatter(fit_quantiles[index], dist.ppf(quantiles), color='C3', marker='+', label='quantiles setpoints')
    ax.axvline(sig[index], color='C3', linestyle='--', label='value to transform')
    ax.plot(sig_values, linear_interp(sig_values), label='linear transform')
    ax.plot(
        extrap_values,
        extrap_ys,
        linestyle='--',
        color='black',
        label='extrapolation transform'
        )
    ax.scatter(sig[index], transf[index], color='black', marker='x', label='transformed value')
    ax.set_title(f'Transfer function from dilated signal to normal distribution - {key} tail')
    ax.legend()
    return ax