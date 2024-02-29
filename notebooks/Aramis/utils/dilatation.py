import utils.interpolation as interpolation
import numpy as np

def build_original_idx(signal_series, nvals_ori):
    # Building a float index from 00:00 first day to last measure of last day (eg. 23:55)
    # In addition, last bin end (00:00 next day)
    # Original signal length + 1
    signal_idx_ori = signal_series.index.astype(int).to_numpy()
    signal_idx_ori = (signal_idx_ori - signal_idx_ori[0])
    signal_idx_ori = signal_idx_ori / signal_idx_ori[1] * 24 / nvals_ori
    dt = signal_idx_ori[-1] - signal_idx_ori[-2]
    signal_idx_ori = np.append(signal_idx_ori, signal_idx_ori[-1] + dt) # Last bin end
    return signal_idx_ori

def build_dilated_idx(sunrises, sunsets, signal_idx_ori, nvals_dil=101):
    # Building a float index from 00:00 first day to last sunset of last day (eg. 18:37)
    # In addition, last bin end (00:00 next day)
    # Dilated signal length + 1
    sunrise_idx_ori = sunrises + 24*np.arange(len(sunrises))
    sunset_idx_ori = sunsets + 24*np.arange(len(sunsets))
    signal_idx_dil = np.linspace(sunrise_idx_ori, sunset_idx_ori, nvals_dil).ravel(order='F')
    signal_idx_dil = np.append(0, signal_idx_dil) # Adding first midnight
    signal_idx_dil = np.append(signal_idx_dil, signal_idx_ori[-1]) # Last bin end
    return signal_idx_dil

def add_night_values(signal_idx_dil, quantiles_dil, nvals_dil):
    ndays = len(signal_idx_dil) // nvals_dil
    matrix = np.zeros((nvals_dil + 1, ndays))
    # Extrapolating the index
    signal_idx_dil_night = np.zeros((nvals_dil + 1) * ndays + 2)
    matrix[:-1] = signal_idx_dil.reshape((nvals_dil, ndays), order='F')
    matrix[-1] = matrix[-2] + (matrix[-2] - matrix[-3]) # Adding night value every day
    signal_idx_dil_night[1:-1] = matrix.ravel(order='F')
    signal_idx_dil_night[-1] = signal_idx_dil_night[-2] + 24 # Closing the last day by adding a false night value
    signal_idx_dil_night[0] = signal_idx_dil_night[1] - 24
    # Extrapolating the signal
    quantiles_dil_night = np.zeros(((nvals_dil + 1) * ndays + 2, quantiles_dil.shape[1]))
    for i in range(quantiles_dil.shape[1]):
        matrix[:-1] = quantiles_dil[:, i].reshape((nvals_dil, ndays), order='F')
        matrix[-1] = 0
        quantiles_dil_night[1:-1, i] = matrix.ravel(order='F')
    quantiles_dil_night[0] = 0
    quantiles_dil_night[-1] = 0
    return signal_idx_dil_night, quantiles_dil_night

def dilate_signal(signal_idx_dil, signal_idx_ori, signal_ori):
    # Dilated index length - 1
    _signal_ori = np.append(signal_ori, signal_ori[-1]) # Adding last dummy value to interpolate
    signal_dil = interpolation.interpolate(signal_idx_dil, signal_idx_ori, _signal_ori, alignment='left')
    return signal_dil

def undilate_signal(signal_idx_ori, signal_idx_dil, signal_dil):
    # Original index length - 1
    _signal_dil = np.append(signal_dil, signal_dil[-1]) # Adding last dummy value to interpolate
    signal_ori = interpolation.interpolate(signal_idx_ori, signal_idx_dil, _signal_dil, alignment='left')
    return signal_ori

def extrapolate_signal_after_sunset(signal, nvals, ndays, method):
    # Signal has length nvals * ndays + 2
    matrix = np.zeros((nvals + 1, ndays))
    matrix[:-1] = signal[1:-1].reshape((nvals, ndays), order='F')
    if method == 'linear':
        matrix[-1] = matrix[-2] + (matrix[-2] - matrix[-3])
    elif method == 'zero_padding':
        matrix[-1] = 0
    else:
        raise ValueError("Invalid value for method. Choose from: ['linear', 'zero_padding']")
    new_signal = np.zeros((nvals + 1) * ndays + 2)
    new_signal[0] = signal[0]
    new_signal[1:-1] = matrix.ravel(order='F')
    new_signal[-1] = signal[-1]
    return new_signal

def undilate_quantiles(signal_idx_ori, signal_idx_dil, quantiles_dil, nvals_dil=101):
    ndays = (len(signal_idx_dil) - 2) // nvals_dil
    new_signal_idx_dil = extrapolate_signal_after_sunset(signal_idx_dil, nvals_dil, ndays, method='linear')
    _quantile_dil = np.zeros(nvals_dil * ndays + 2)
    
    quantiles_ori = np.zeros((signal_idx_ori.shape[0]-1, quantiles_dil.shape[1]))
    for i in range(quantiles_dil.shape[1]):
        _quantile_dil[:-1] = quantiles_dil[:,i]
        new_quantile_dil = extrapolate_signal_after_sunset(_quantile_dil, nvals_dil, ndays, method='zero_padding')
        quantiles_ori[:,i] = interpolation.interpolate(signal_idx_ori, new_signal_idx_dil, new_quantile_dil, alignment='left')

    return quantiles_ori

