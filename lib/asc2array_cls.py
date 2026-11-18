import os

import numpy as np
from scipy.stats import norm

from au2mm import au2mm
from band_pass_filter import lowpass_filter
from pre_processing import re_sampling
from zeroInterp import adaptivePad, zeroInterp


class asc2array_cls:
    """EyeLink .asc -> pupil / gaze arrays, blink rejection and interpolation"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.s_trg = cfg["s_trg"] if len(cfg["s_trg"]) > 0 else "Start_Experiment"

    def dataExtraction(self, fileName, encode="utf-8"):
        with open(fileName, encoding=encode) as f:
            return [line.split() for line in f.readlines()]

    def dataParse(self, dat):

        print(f"Analysing...{self.cfg['fName']}")

        eyeData = {eye: {"timestamp": [], "gazeX": [], "gazeY": [], "pupil": []} for eye in ["Right", "Left"]}
        events = {'SFIX': [], 'EFIX': [], 'SSACC': [], 'ESACC': [], 'SBLINK': [], 'EBLINK': [], 'MSG': []}

        for line in dat:
            if '!MODE' in line:
                eyeRecording = line[-1]
                break

        def value(s):
            return float(s) if s.replace('.', '', 1).isdigit() else 0

        start = False
        for line in dat:
            if start:
                if len(line) > 3 and line[0].isdecimal():
                    eyeData['Left']["timestamp"].append(float(line[0]))
                    eyeData['Left']["gazeX"].append(value(line[1]))
                    eyeData['Left']["gazeY"].append(value(line[2]))
                    eyeData['Left']["pupil"].append(value(line[3]))

                    eyeData['Right']["timestamp"].append(float(line[0]))
                    if eyeRecording == "LR":
                        eyeData['Right']["gazeX"].append(value(line[4]))
                        eyeData['Right']["gazeY"].append(value(line[5]))
                        eyeData['Right']["pupil"].append(value(line[6]))
                    else:
                        for mmName in ["gazeX", "gazeY", "pupil"]:
                            eyeData['Right'][mmName].append(eyeData['Left'][mmName][-1])

                if len(line) == 0:
                    continue

                if line[0] in events:
                    events[line[0]].append(line[1:])
            else:
                if 'PUPIL' in line:
                    self.Method = line[1]

                elif 'RATE' in line:
                    for l in line:
                        if l.replace('.', '', 1).isdigit():
                            self.fs = float(l)
                            break

                if self.s_trg in line:
                    start = True
                    self.initialTimeVal = int(line[1])
                    if self.s_trg != 'Start_Experiment':
                        events['MSG'].append(line[1:])

        ## samples on a regular time axis (0 = missing)
        timeStamp = {iEyes: [round((t - self.initialTimeVal)*(self.fs/1000)) for t in eyeData[iEyes]["timestamp"]]
                     for iEyes in ["Left", "Right"]}
        timeLen = max(np.max(timeStamp["Left"]), np.max(timeStamp["Right"]))

        for mmName in ['gazeX', 'gazeY', 'pupil']:
            eyeData[mmName] = np.zeros((2, timeLen+1))
            for iEye, eye in enumerate(["Left", "Right"]):
                eyeData[mmName][iEye, timeStamp[eye]] = eyeData[eye][mmName]

        ## blinkInterp() works on pupil down-sampled to cfg["INTERP_FS"] Hz (default 250)
        self.dataLen = eyeData["pupil"].shape[1]
        interp_fs = self.cfg.get("INTERP_FS", 250)
        if self.fs > interp_fs:
            eyeData["pupil"] = re_sampling(eyeData["pupil"].copy(), round(self.dataLen*(interp_fs/self.fs)))

        ## samples faster than the dilation-speed threshold are set to 0
        pupil_fs = interp_fs if self.fs > interp_fs else self.fs
        eyeData["pupilSpeed"] = [self.pupilSpeed(eyeData["pupil"][iEyes, :], iEyes, pupil_fs)
                                 for iEyes in np.arange(eyeData["pupil"].shape[0])]
        th = self.cfg.get("PUPIL_SPEED_TH")
        if th is None:
            th = [self.speedThreshold([s], self.cfg.get("PUPIL_SPEED_MAD_N", 16)) for s in eyeData["pupilSpeed"]]
        for iEyes in np.arange(eyeData["pupil"].shape[0]):
            ind = np.argwhere(eyeData["pupilSpeed"][iEyes] > th[iEyes]).reshape(-1)
            eyeData["pupil"][iEyes, ind] = 0

        ## gaze: jumps outside the 99% interval of the sample-to-sample change are set to 0
        for mmName in ["gazeX", "gazeY"]:
            for iEyes in np.arange(eyeData["pupil"].shape[0]):
                d = np.diff(eyeData[mmName][iEyes, :])
                q75, q25 = np.percentile(d, [75, 25])
                IQR = q75 - q25
                d_witout_outliar = d[(d > q25 - IQR*1.5) & (d < q75 + IQR*1.5)]
                param = norm.fit(d_witout_outliar)
                (a, b) = norm.interval(0.99, loc=param[0], scale=param[1])

                ind = np.argwhere(abs(np.diff(eyeData[mmName][iEyes, :])) > b).reshape(-1)
                eyeData[mmName][iEyes, ind] = 0

        return eyeData, events, self.initialTimeVal, self.fs

    def pupilSpeed(self, pupil, iEyes, fs):
        """Dilation speed [mm/s] of one eye (Kret & Sjak-Shie, 2019)"""
        pupil = np.asarray(pupil, dtype=float)
        x = au2mm(pupil.copy(), self.Method, "LR"[iEyes], vis_dist=self.cfg.get("VISUAL_DISTANCE"))
        x[pupil == 0] = np.nan
        d = np.abs(np.diff(x)) * fs
        return np.fmax(np.r_[np.nan, d], np.r_[d, np.nan])

    @staticmethod
    def speedThreshold(speed, n=16):
        """median + n * MAD of the dilation speeds in the list `speed` (pooled)"""
        s = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in speed])
        s = s[np.isfinite(s)]
        if s.size == 0:
            return float("inf")
        med = np.median(s)
        return float(med + n * np.median(np.abs(s - med)))

    def pupilSpeedThreshold(self, ascFiles):
        """Per-participant pupil speed threshold [left, right] in mm/s, all runs pooled"""
        speed = [[], []]
        for ascFile in ascFiles:
            t2a = asc2array_cls({**self.cfg, "PUPIL_SPEED_TH": [np.inf, np.inf],
                                 "fName": os.path.splitext(os.path.basename(ascFile))[0]})
            eyeData, _, _, _ = t2a.dataParse(t2a.dataExtraction(ascFile))
            for iEyes, s in enumerate(eyeData["pupilSpeed"]):
                speed[iEyes].append(s)
        return [self.speedThreshold(s, self.cfg.get("PUPIL_SPEED_MAD_N", 16)) for s in speed]

    def blinkInterp(self, eyeData):
        """Blink padding and PCHIP interpolation, then the eye with fewer interpolated samples is used"""

        pupilData = eyeData["pupil"]
        pupil_fs = self.fs if pupilData.shape[1] == self.dataLen else self.cfg.get("INTERP_FS", 250)

        pupilData_lowpass = pupilData.copy()
        for iEyes in np.arange(2):
            pupilData_lowpass[iEyes, pupilData_lowpass[iEyes, :] != 0] = \
                lowpass_filter(pupilData[iEyes, pupilData[iEyes, :] != 0], 40, pupil_fs)
        pupilData_lowpass[pupilData_lowpass < 0] = 0

        interval = pupil_fs*self.cfg.get("BLINK_INTERVAL", 0.1)
        padLen = round(pupil_fs*self.cfg.get("BLINK_PAD", 0))
        padMinLen = round(pupil_fs*self.cfg.get("BLINK_PAD_MINLEN", 0))

        pupilData_pad = pupilData_lowpass.copy()
        for iEyes in np.arange(2):
            x = au2mm(pupilData_lowpass[iEyes, :].copy(), self.Method, "LR"[iEyes],
                      vis_dist=self.cfg.get("VISUAL_DISTANCE")).reshape(-1)
            x[pupilData_lowpass[iEyes, :] == 0] = 0
            m = adaptivePad(x, pupil_fs, interval, padLen=padLen, padMinLen=padMinLen,
                            speedTh=self.cfg["BLINK_PAD_ADAPTIVE_V"],
                            maxPre=round(pupil_fs*self.cfg.get("BLINK_PAD_MAX_PRE", 0.1)),
                            maxPost=round(pupil_fs*self.cfg.get("BLINK_PAD_MAX_POST", 0.3)),
                            smoothLen=round(pupil_fs*0.05),
                            minIsland=round(pupil_fs*self.cfg.get("BLINK_MIN_ISLAND", 0.1)))
            pupilData_pad[iEyes, m] = 0
        pupilData_intep = zeroInterp(pupilData_pad, interval)

        zeroArray = np.zeros((2, pupilData.shape[1]))
        for iEyes in np.arange(2):
            zeroArray[iEyes, pupilData_intep["zeroArray"][iEyes]] = 1

        iEyes = 0 if pupilData_intep['interpolatedArray'][0] < pupilData_intep['interpolatedArray'][1] else 1
        pupilData = pupilData_intep['pupilData'][[iEyes], :]
        gazeX = eyeData["gazeX"][[iEyes], :]
        gazeY = eyeData["gazeY"][[iEyes], :]
        zeroArray = zeroArray[[iEyes], :]

        ## back to the original sampling rate
        pupilData = re_sampling(pupilData.copy(), self.dataLen)
        if zeroArray.shape[1] != pupilData.shape[1]:
            source_index = np.minimum(
                (np.arange(pupilData.shape[1]) * zeroArray.shape[1] / pupilData.shape[1]).astype(int),
                zeroArray.shape[1] - 1,
            )
            zeroArray = zeroArray[:, source_index]

        ## pupilData is returned in a.u.
        return {'pupilData': pupilData,
                'gazeX': gazeX,
                'gazeY': gazeY,
                "zeroArray": zeroArray,
                'usedEye': "LR"[iEyes]}
