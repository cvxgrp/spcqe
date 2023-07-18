import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
from math import pi
import time
import scipy as sp
from scipy import stats
from scipy.stats import norm


def individual_bases_3(K, P1, P2, P3):
    w1 = 2*pi/P1
    w2 = 2*pi/P2
    w3 = 2*pi/P3
    Base = np.zeros((P1, 3*(2*K)+1))
    for t in range(P1):
        Basis = [1]
        for i in range(1, K+1):
            Basis.append(math.cos(i*w1*t))
            Basis.append(math.sin(i*w1*t))
        for i in range(1, K+1):
            Basis.append(math.cos(i*w2*t))
            Basis.append(math.sin(i*w2*t))
        for i in range(1, K+1):
            Basis.append(math.cos(i*w3*t))
            Basis.append(math.sin(i*w3*t))
        Base[t, :] = Basis
    B_P0 = Base[:, 0].reshape(-1, 1)
    B_P1 = Base[:, 1:2*K+1]
    B_P2 = Base[:, 2*K+1:2*2*K+1]
    B_P3 = Base[:, 2*2*K+1:2*2*2*K+1]
    Bases = [B_P0, B_P1, B_P2, B_P3]
    return Bases  # (P1,3*2K+1)


def individual_bases_2(K, P1, P3):
    w1 = 2*pi/P1
    w3 = 2*pi/P3
    Base = np.zeros((P1, 2*(2*K)+1))
    for t in range(P1):
        Basis = [1]
        for i in range(1, K+1):
            Basis.append(math.cos(i*w1*t))
            Basis.append(math.sin(i*w1*t))
        for i in range(1, K+1):
            Basis.append(math.cos(i*w3*t))
            Basis.append(math.sin(i*w3*t))
        Base[t, :] = Basis
    B_P0 = Base[:, 0].reshape(-1, 1)
    B_P1 = Base[:, 1:2*K+1]
    B_P3 = Base[:, 2*K+1:2*2*K+1]
    Bases = [B_P0, B_P1, B_P3]
    return Bases  # (P1,2*2K+1)


def cross_bases(B_P1, B_P2):
    # Compute the outer products for each pair of corresponding rows
    outer_products = [np.outer(B_P1[i], B_P2[i]).flatten(
    ).reshape(-1, 1) for i in range(B_P1.shape[0])]
    # Concatenate along the first axis to create a 2D array
    result = np.concatenate(outer_products, axis=1).T
    return result


def basis_3(K, P1, P2, P3):
    Bases = individual_bases_3(K, P1, P2, P3)
    B_P0 = Bases[0]
    B_P1 = Bases[1]
    B_P2 = Bases[2]
    B_P3 = Bases[3]
    C_12 = cross_bases(B_P1, B_P2)
    C_13 = cross_bases(B_P1, B_P3)
    C_23 = cross_bases(B_P2, B_P3)
    B = np.hstack([B_P0, B_P1, B_P2, B_P3, C_12, C_13, C_23])
    return B


def basis_2(K, P1, P3):
    Bases = individual_bases_2(K, P1, P3)
    B_P0 = Bases[0]
    B_P1 = Bases[1]
    B_P3 = Bases[2]
    C_13 = cross_bases(B_P1, B_P3)
    B = np.hstack([B_P0, B_P1, B_P3, C_13])
    return B


def regularization_matrix_3(K, l, P1, P2, P3):
    l1 = l*(2*np.pi)/np.sqrt(P1)
    l2 = l*(2*np.pi)/np.sqrt(P2)
    l3 = l*(2*np.pi)/np.sqrt(P3)
    l4 = l*(2*np.pi)/np.sqrt(P2)
    l5 = l*(2*np.pi)/np.sqrt(P3)
    l6 = l*(2*np.pi)/np.sqrt(P3)
    # Do not penalize 0'th Fourier coefficient corresponding to DC signal.
    coeff_i = [0]
    for i in range(1, K+1):
        coeff_i.append(l1*i)
        coeff_i.append(l1*i)
    for i in range(1, K+1):
        coeff_i.append(l2*i)
        coeff_i.append(l2*i)
    for i in range(1, K+1):
        coeff_i.append(l3*i)
        coeff_i.append(l3*i)
    for j in range(0, 2*K):
        for i in range(1, K+1):
            coeff_i.append(l4*i)
            coeff_i.append(l4*i)
    for j in range(0, 2*K):
        for i in range(1, K+1):
            coeff_i.append(l5*i)
            coeff_i.append(l5*i)
    for j in range(0, 2*K):
        for i in range(1, K+1):
            coeff_i.append(l6*i)
            coeff_i.append(l6*i)
    coeff_i = np.array(coeff_i)
    D = np.diag(coeff_i)  # [0,1,1,4,4,9,9,...,K^2,K^2] diagonal matrix
    return D


def regularization_matrix_2(K, l, P1, P3):
    l1 = l*(2*np.pi)/np.sqrt(P1)
    l3 = l*(2*np.pi)/np.sqrt(P3)
    l5 = l*(2*np.pi)/np.sqrt(P3)
    # Do not penalize 0'th Fourier coefficient corresponding to DC signal.
    coeff_i = [0]
    for i in range(1, K+1):
        coeff_i.append(l1*i)
        coeff_i.append(l1*i)
    for i in range(1, K+1):
        coeff_i.append(l3*i)
        coeff_i.append(l3*i)
    for j in range(0, 2*K):
        for i in range(1, K+1):
            coeff_i.append(l5*i)
            coeff_i.append(l5*i)
    coeff_i = np.array(coeff_i)
    D = np.diag(coeff_i)  # [0,1,1,4,4,9,9,...,K^2,K^2] diagonal matrix
    return D


def pinball_slopes(percentiles):
    percentiles = np.asarray(percentiles)
    a = (percentiles-50)*(0.01)
    b = (0.5)*np.ones((len(a),))
    return a, b


def fit_quantiles(y1, B, D, percentiles):
    num_years = y1.shape[1]
    a, b = pinball_slopes(percentiles)
    num_quantiles = len(a)
    Theta = cp.Variable((B.shape[1], num_quantiles))
    BT = B@Theta
    obj = 0
    for i in range(num_years):
        nonnanindex = ~np.isnan(y1[:, i])
        Var = y1[nonnanindex, i].reshape(-1, 1) - BT[nonnanindex]
        obj += cp.sum(Var@np.diag(a)+cp.abs(Var)@np.diag(b))
    # ensures quantiles are in order and prevents point masses ie minimum distance.
    cons = [cp.diff(BT, axis=1) >= 0.01]
    # cons+=[BT[:,0]>=0] #ensures quantiles are nonnegative
    Z = D @ Theta
    regularization = cp.sum_squares(Z)
    obj += regularization
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(verbose=True, solver=cp.MOSEK)
    Q = B@Theta.value
    # --------------------
    Q_left_tail = Q[:, 0]-1
    Q_left_tail = Q_left_tail.reshape(-1, 1)
    Q_right_tail = Q[:, -1]+1
    Q_right_tail = Q_right_tail.reshape(-1, 1)
    Q_extended = np.hstack([Q_left_tail, Q, Q_right_tail])
    # --------------------
    percentiles = np.asarray(percentiles)
    percentiles = percentiles/100
    g = np.zeros((Q_extended.shape[0], len(percentiles)+2))
    g[:, 1:-1] = stats.norm.ppf(percentiles)
    # --------------------
    left_tail_slope = 1*(g[:, 2]-g[:, 1])/(Q_extended[:, 2]-Q_extended[:, 1])
    right_tail_slope = 1*(g[:, -2]-g[:, -3]) / \
        (Q_extended[:, -2]-Q_extended[:, -3])
    g[:, 0] = g[:, 1]-left_tail_slope
    g[:, -1] = g[:, -2]+right_tail_slope
    GQ_extended = g
    return Q, Q_extended, GQ_extended
