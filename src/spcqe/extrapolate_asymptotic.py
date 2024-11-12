""" Asymptotic Extrapolation Module

This module is for extrapolating the tails of a signal using asymptotic functions.
Output space asymptotes have the form
                yasympt + alpha * exp(beta * x)
Input space asymptotes have the form
                beta * np.log(alpha * (x - xasympt))
It also provide plot functions for the usecase of pv signal.
                
Author: Aramis Dufour
"""


import numpy as np
import scipy.stats as sps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

dist = sps.norm()
XSOLAR = 0
YSOLAR = dist.ppf(0.99999)

def get_asymptote_parameters_out(x0, _y0, x1, _y1, yasympt):
    """
    Get the parameters of the asymptote function in the output
    space :
    yasympt + alpha * exp(beta * x)
    with continuity of the function and its first derivative
    at (x0, y0).
    """
    y0, y1 = dist.ppf(_y0), dist.ppf(_y1)
    yprime0 = (y0 - y1) / (x0 - x1)
    beta = safe_divide(yprime0, y0 - yasympt)
    exp_input = -beta * x0
    exp_output = safe_exp(exp_input)
    alpha = - (yasympt - y0) * exp_output
    return alpha, beta

def get_asymptote_parameters_in(x0, _y0, x1, _y1, xasympt):
    """
    Get the parameters of the asymptote function in the input
    space :
    beta * np.log(alpha * (x - xasympt))
    with continuity of the function and its first derivative
    at (x0, y0).
    """
    y0, y1 = dist.ppf(_y0), dist.ppf(_y1)
    yprime0 = (y1 - y0) / (x1 - x0)
    beta = yprime0 * (x0 - xasympt)
    exp_input = y0 / beta
    exp_output = safe_exp(exp_input)
    alpha = 1 / (x0 - xasympt) * exp_output
    return alpha, beta

def asymptote_out(x, yasympt, alpha, beta):
    """
    Transform with an aymptote in the output
    space.
    """
    exp_input = beta * x
    exp_output = safe_exp(exp_input)
    return yasympt + alpha * exp_output


def asymptote_in(x, xasympt, alpha, beta):
    """
    Transform with an aymptote in the input
    space.
    """
    log_input = alpha * (x - xasympt)
    log_output = safe_log(log_input)
    return beta * log_output

def inverse_asymptote_out(y, yasympt, alpha, beta):
    """
    Inverse transform with an aymptote in the
    output space.
    """
    log_input = safe_divide(y - yasympt, alpha)
    log_output = safe_log(log_input)
    return safe_divide(log_output, beta)

def inverse_asymptote_in(y, xasympt, alpha, beta):
    """
    Inverse transform with an aymptote in the
    input space.
    """
    exp_input = safe_divide(y, beta)
    exp_output = safe_exp(exp_input)
    return xasympt + safe_divide(exp_output, alpha)

def safe_exp(exp_input):
    """
    Safe exponential function.
    Avoids overflow and underflow with float64.
    """
    exp_output = np.empty_like(exp_input)
    exp_output[exp_input > 709] = np.inf
    exp_output[exp_input < -709] = 0
    exp_output[(-709 <= exp_input) & (exp_input <= 709)] = np.exp(exp_input[(-709 <= exp_input) & (exp_input <= 709)])
    return exp_output

def safe_log(log_input):
    """
    Safe logarithm function.
    Avoids forbidden operations.
    """
    log_output = np.empty_like(log_input)
    log_output[log_input <= 0] = -np.inf
    log_output[log_input > 0] = np.log(log_input[log_input > 0])
    return log_output

def safe_divide(a, b):
    """
    Safe division.
    Avoids division by zero.
    """
    division_output = np.empty_like(a)
    division_output[np.logical_and(a==0, b==0)] = np.nan
    division_output[np.logical_and(a!=0, b==0)] = a[np.logical_and(a!=0, b==0)] * np.inf
    division_output[b != 0] = a[b != 0] / b[b != 0]
    return division_output

def init_extrap(extrapolate):
    """
    Initialize the extrapolation parameters
    when using preset keywords.
    """
    if extrapolate == None:
        return init_extrap('linear')
    if extrapolate == 'linear':
        return {'lower': 'linear', 'upper': 'linear'}
    elif extrapolate == 'solar':
        extrapolate = {'lower':(XSOLAR, 'input'), 'upper':(YSOLAR, 'output')}
    return extrapolate


def plot_pdf(ax, transf, label):
    """
    Plot the transformed signal pdf and the
    normal pdf.
    """
    x = np.linspace(-4, 4, 1000)
    y = sps.norm.pdf(x)
    sns.histplot(transf, bins=1000, stat='density', ax=ax, kde=False, label=label)
    ax.plot(x, y, 'r', label='Normal PDF')
    ax.set_xlabel('Transformed Signal')
    ax.legend()
    return ax

def plot_tails(ax, sig, quantiles, fit_quantiles, transf, method, key, index, extrap_width,
               h_per_day=None,
               params=None,
               asymptote=None,
               space=None,
               n_days=15,
               n_hours=3,):
    """
    Plot the extrapolation of the tails of the signal.
    As an option if h_per_day is not None, will plot
    the transformed values of the nearby days and hours.
    """
    linear_interp = interp1d(fit_quantiles[index], dist.ppf(quantiles), kind='linear', fill_value='extrapolate')
    if key == 'upper':
        sig_m, sig_M = fit_quantiles[index, 0], sig[index]
        extrap_values = np.linspace(fit_quantiles[index, -1], fit_quantiles[index, -1] + extrap_width, 1000)
        if method == 'asymptotic':
            if space == 'output':
                extrap_ys = asymptote_out(extrap_values, asymptote, params[0][index], params[1][index])
            elif space == 'input':
                extrap_ys = asymptote_in(extrap_values, asymptote, params[0][index], params[1][index])
        elif method == 'linear':
            extrap_ys = linear_interp(extrap_values)
    elif key == 'lower':
        sig_m, sig_M = sig[index], fit_quantiles[index, -1]
        extrap_values = np.linspace(fit_quantiles[index, 0] - extrap_width, fit_quantiles[index, 0], 1000)
        if method == 'asymptotic':
            if space == 'output':
                extrap_ys = asymptote_out(extrap_values, asymptote, params[0][index], params[1][index])
            elif space == 'input':
                extrap_ys = asymptote_in(extrap_values, asymptote, params[0][index], params[1][index])
        elif method == 'linear':
            extrap_ys = linear_interp(extrap_values)
    sig_values = np.linspace(sig_m, sig_M, 1000)
    ax.scatter(fit_quantiles[index], dist.ppf(quantiles), color='C3', marker='+', s=40, label='quantiles setpoints')
    if h_per_day is not None:
        idxs_days = np.arange(-n_days, n_days+1, 1) * h_per_day + index
        idxs_days = np.array([i for i in idxs_days if i != index])
        idxs_hours = np.arange(-n_hours, n_hours+1, 1) + index
        idxs_hours = np.array([i for i in idxs_hours if i != index])
        ax.scatter(sig[idxs_days], transf[idxs_days], color='green', s=20, label='nearby days')
        ax.scatter(sig[idxs_hours], transf[idxs_hours], color='orange', s=20, label='nearby hours')
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

def get_tail(X, quantiles, tail):
    """
    Get the tail of the signal.
    """
    if tail == 'upper':
        mask = X > dist.ppf(quantiles[-1])
    elif tail == 'lower':
        mask = X < dist.ppf(quantiles[0])
    return mask