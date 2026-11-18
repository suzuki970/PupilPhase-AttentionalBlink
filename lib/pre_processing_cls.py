import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class pre_processing:

    def __init__(self, cfg):
        self.cfg = cfg

    def pre_processing(self, dat):
        """Baseline subtraction (cfg["WID_BASELINE"]) and artifact rejection (cfg["THRES_DIFF"])
        of trials x samples data spanning cfg["TIME_START"] to cfg["TIME_END"]"""
        y = dat.copy()
        x = np.linspace(self.cfg['TIME_START'], self.cfg['TIME_END'], y.shape[1])

        baselineData = np.array([getNearestValue(x, self.cfg['WID_BASELINE'][0][0]),
                                 getNearestValue(x, self.cfg['WID_BASELINE'][0][1]),
                                 getNearestValue(x, self.cfg['WID_ANALYSIS'])])
        baselinePLR = np.mean(y[:, np.arange(baselineData[0], baselineData[1])], axis=1)
        y = y - baselinePLR[:, np.newaxis]

        rejectNum = reject_trials(y, self.cfg['THRES_DIFF'], baselineData)

        return y, rejectNum


def rejectDat(dat, rejectNum):
    """Remove the trials in rejectNum from every list in dat"""
    for mm in list(dat.keys()):
        dat[mm] = [d for i, d in enumerate(dat[mm]) if i not in rejectNum]
    return dat


def getNearestValue(in_y, num):
    return np.abs(np.asarray(in_y) - num).argmin()


def reject_trials(y, thres, baselineData):
    """Trials with a sample-to-sample change > thres, or with many zero / NaN samples"""

    fx = abs(np.diff(y, n=1))
    fx = fx[:, np.arange(baselineData[0], baselineData[2])]
    rejectNum = np.unique(np.argwhere(fx > thres)[:, 0])

    rejectNum_zero = []
    for i in np.arange(y.shape[0]):
        if sum(np.argwhere(y[i, np.arange(baselineData[0], baselineData[2])] == 0)) > y.shape[1] / 2:
            rejectNum_zero.append(int(i))
    rejectNum = np.r_[rejectNum, np.array(rejectNum_zero)]

    rejectNum_nan = []
    for i in np.arange(y.shape[0]):
        if len(np.argwhere(np.isnan(y[i, np.arange(baselineData[0], baselineData[2])]) == True)) > y.shape[1] / 2:
            rejectNum_nan.append(int(i))
    rejectNum = np.r_[rejectNum, np.array(rejectNum_nan)]

    return np.unique(rejectNum).astype("int64").tolist()


def rejectedByOutlier(dat, y_bp, factorName="BaselinePD"):
    """Trials whose y_bp is outside [Q1 - 3 IQR, Q3 + 3 IQR] of each participant"""

    tmp = pd.DataFrame()
    tmp["sub"] = np.array(dat["sub"])
    tmp[factorName] = y_bp

    q75 = tmp.groupby(["sub"], as_index=False).quantile(.75, numeric_only=True)
    q25 = tmp.groupby(["sub"], as_index=False).quantile(.25, numeric_only=True)
    IQR = q75[factorName] - q25[factorName]

    lower = q25
    lower[factorName] = lower[factorName] - IQR*3
    upper = q75
    upper[factorName] = upper[factorName] + IQR*3

    tmp["lower"] = 0
    tmp["upper"] = 0
    for iSub in np.unique(tmp["sub"]):
        tmp.loc[tmp["sub"] == iSub, "lower"] = float(lower[factorName][lower["sub"] == iSub].values)
        tmp.loc[tmp["sub"] == iSub, "upper"] = float(upper[factorName][lower["sub"] == iSub].values)

    reject = tmp[(tmp[factorName] < tmp["lower"]) | (tmp[factorName] > tmp["upper"])].index.values
    return reject.tolist()


def getPCPDevents(y, minHeight):
    """Pupil peaks (onset of the PC phase) and troughs (onset of the PD phase) of each row,
    detected with scipy.signal.find_peaks(prominence=minHeight)"""

    if not isinstance(y, list):
        if y.ndim == 1:
            y = y.reshape(1, len(y))
        len_y = y.shape[1]

    events = {'peaks': [], 'troughs': []}
    for iTrial in np.arange(len(y)):
        peaks, _ = find_peaks(y[iTrial].reshape(-1), prominence=minHeight)
        troughs, _ = find_peaks(-y[iTrial].reshape(-1), prominence=minHeight)
        events["peaks"].append(peaks.tolist())
        events["troughs"].append(troughs.tolist())

    ## 1 = peak, -1 = trough
    if isinstance(y, list):
        events["summary"] = []
        for iRun, (peak, trough) in enumerate(zip(events["peaks"], events["troughs"])):
            tmp_summary = np.zeros((y[iRun].shape[1])).astype(int)
            tmp_summary[peak] = 1
            tmp_summary[trough] = -1
            events["summary"].append(tmp_summary.tolist())
    else:
        events["summary"] = np.zeros((len(y), len_y)).astype(int)
        for iRun, (peak, trough) in enumerate(zip(events["peaks"], events["troughs"])):
            events["summary"][iRun, peak] = 1
            events["summary"][iRun, trough] = -1

    return events
