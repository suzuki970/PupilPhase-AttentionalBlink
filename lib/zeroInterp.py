import numpy as np
from scipy import interpolate


def zeroInterp(pupilData, interval):
    """PCHIP interpolation of the zero (missing) samples of each row.
    Zero runs closer than `interval` samples are merged into one gap."""

    if pupilData.ndim == 1:
        pupilData = pupilData.reshape(1, len(pupilData))

    interpolatedArray = []
    zeroArray = []

    for iTrial, y_base in enumerate(list(pupilData)):
        y_base = np.array(y_base)
        y = y_base.copy()
        y[np.argwhere(y < 10**-5)] = 0

        zeroInd = np.argwhere(y == 0).reshape(-1)

        ## missing samples at the edges take the nearest valid value
        if 0 in zeroInd:
            for iSample in np.arange(len(y)):
                if y[iSample] != 0:
                    y[:iSample-1] = y[iSample]
                    y_base[:iSample-1] = y_base[iSample]
                    break

        if len(y)-1 in zeroInd:
            for iSample in np.arange(1, len(y)):
                if y[-iSample] != 0:
                    y[-iSample+1:] = y[-iSample]
                    y_base[-iSample+1:] = y_base[-iSample]
                    break

        zeroInd = np.argwhere(y[0:-1] == 0).reshape(-1)

        if len(zeroInd) == len(y)-1:
            zeroArray.append(np.arange(len(y)))
            interpolatedArray.append(len(y))
            continue

        if len(zeroInd) > 0:

            diffOnOff = np.diff(zeroInd)
            diffOnOff = np.append(diffOnOff, 10**10)
            diffOnOff = np.append(10**10, diffOnOff)

            ## onset and offset of each gap
            count = 0
            datOfblinkCood = []
            for i in np.arange(1, len(diffOnOff)):
                if diffOnOff[i] >= interval and diffOnOff[i - 1] >= interval:
                    datOfblinkCood.append(np.array([zeroInd[i - 1], zeroInd[i - 1]]))
                    count = count + 1
                elif diffOnOff[i] >= interval and diffOnOff[i - 1] <= interval:
                    datOfblinkCood[count][1] = zeroInd[i - 1]
                    count = count + 1
                elif diffOnOff[i] < interval and diffOnOff[i - 1] >= interval:
                    datOfblinkCood.append(np.array([zeroInd[i - 1], 0]))

            for onsetArray, offsetArray in datOfblinkCood:
                if onsetArray == offsetArray:
                    y_base[onsetArray] = 0
                else:
                    y_base[onsetArray:offsetArray] = 0

            y = y_base.copy()
            nonZero = np.argwhere(y != 0).reshape(-1)
            zeroArray.append(np.argwhere(y == 0).reshape(-1))

            if len(nonZero) != len(y):
                numX = np.arange(len(y))
                yy = interpolate.PchipInterpolator(numX[nonZero], y[nonZero], extrapolate=False)
                interp = yy(np.arange(len(y)))
                lo, hi = nonZero[0], nonZero[-1]
                interp[:lo] = y[lo]
                interp[hi + 1:] = y[hi]
                pupilData[iTrial] = interp

            interpolatedArray.append(len(y)-len(nonZero))

        else:
            interpolatedArray.append(0)

    return {'pupilData': pupilData,
            'interpolatedArray': interpolatedArray,
            'zeroArray': zeroArray}


def adaptivePad(pupil_mm, fs, interval, padLen=0, padMinLen=0, speedTh=1.5,
                maxPre=0, maxPost=0, smoothLen=1, minIsland=0):
    """Blink mask with speed-dependent padding"""
    x = np.asarray(pupil_mm, dtype=float).reshape(-1)
    n = len(x)
    miss = x == 0
    if miss.all() or not miss.any():
        return miss.copy()

    def runs(m, merge):
        d = np.diff(np.r_[0, m.astype(int), 0])
        out = []
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if out and a - out[-1][1] < merge:
                out[-1][1] = b
            else:
                out.append([a, b])
        return out

    ## speed of the smoothed pupil, gaps bridged linearly
    ok = np.flatnonzero(~miss)
    xs = np.interp(np.arange(n), ok, x[ok])
    if smoothLen > 1:
        xs = np.convolve(xs, np.ones(smoothLen) / smoothLen, mode="same")
    v = np.gradient(xs) * fs

    mask = miss.copy()
    for a, b in runs(miss, interval):
        p = padLen if b - a >= padMinLen else 0
        a2, b2 = max(a - p, 0), min(b + p, n)
        lim = max(a2 - maxPre, 0)
        while a2 > lim and v[a2 - 1] < -speedTh:
            a2 -= 1
        lim = min(b2 + maxPost, n)
        while b2 < lim and v[b2] > speedTh:
            b2 += 1
        mask[a2:b2] = True

    for a, b in runs(~mask, 0):
        if b - a < minIsland:
            mask[a:b] = True
    return mask
