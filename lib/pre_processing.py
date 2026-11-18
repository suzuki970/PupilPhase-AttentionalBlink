import numpy as np
from scipy import interpolate


def re_sampling(dat, num, keep_nan=False):
    """PCHIP resampling of each row of dat to num samples.
    keep_nan: output samples next to a NaN input sample are set to NaN"""

    re_sampled = []
    for iTrial, d in enumerate(dat):
        t = np.array(d)
        n = num[iTrial] if isinstance(num, list) else num

        x_new = np.linspace(0, len(t)-1, n)

        x = np.arange(len(t), dtype=float)
        valid = ~np.isnan(t)

        if len(np.argwhere(np.isnan(t))) == len(t):
            re_sampled.append(np.zeros(num)*np.nan)
            continue

        yy = interpolate.PchipInterpolator(x[valid], t[valid], extrapolate=False)
        y_new = yy(x_new)

        xmin, xmax = x[valid][0], x[valid][-1]
        outside = (x_new < xmin) | (x_new > xmax)
        if np.any(outside):
            y_new[outside] = np.nan

        if keep_nan:
            lo = np.floor(x_new).astype(int)
            hi = np.minimum(lo + 1, len(t) - 1)
            y_new[~(valid[lo] & valid[hi])] = np.nan

        re_sampled.append(y_new)

    return np.array(re_sampled)
