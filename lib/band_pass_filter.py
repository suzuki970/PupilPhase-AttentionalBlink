import numpy as np
from scipy import signal


def lowpass_filter(data, highcut, fs, order=4):
    """Zero-phase Butterworth low-pass filter (array, or list of arrays)"""
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = signal.butter(order, high, btype='low', output='sos')
    _pad = 3 * (2 * sos.shape[0] + 1)          # scipy's default sosfiltfilt padlen

    def _apply(d):
        d = np.asarray(d, dtype=float)
        if d.size <= _pad + 1 or not np.all(np.isfinite(d)):
            return d                            # too short / non-finite -> unchanged
        return signal.sosfiltfilt(sos, d, padlen=min(_pad, d.size - 1))

    if isinstance(data, list):
        return [_apply(d).tolist() for d in data]
    return _apply(data)


def nan_lowpass(rows, cutoff, fs):
    """lowpass_filter() of rows with NaN gaps, which are linearly bridged before filtering"""
    filled, missing = [], []
    for r in rows:
        r = np.array(r, dtype=float).reshape(1, -1)
        m = np.isnan(r[0])
        if m.any() and (~m).sum() > 1:
            x = np.arange(r.shape[1])
            r[0, m] = np.interp(x[m], x[~m], r[0, ~m])   # edges: nearest valid value
        filled.append(r)
        missing.append(m)
    lp = lowpass_filter(filled, cutoff, fs)
    return [np.array(f) for f in lp], missing
