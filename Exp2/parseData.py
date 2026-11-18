#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import os
## run from this folder, also when executed from the workspace root (e.g. Positron)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath("Exp2" if os.path.isdir("Exp2") else ".")
os.chdir(HERE)
sys.path.append(os.path.join(HERE, "..", "lib"))
import datetime
import glob
import json
import os
import random

import msgpack
import numpy as np
from joblib import Parallel, delayed

from asc2array_cls import asc2array_cls
from au2mm import au2mm
from pre_processing import re_sampling

dt_now = datetime.datetime.now()
today = dt_now.strftime("%Y%m%d")
date = f"./data/{today}"

folderName = sorted(glob.glob("./rawData/*"))
random.shuffle(folderName)


cfg={
     "RESAMPLING_RATE":200,
     "usedEye":1,
     "WID_FILTER":[],
     "mmFlag":True,
     "VISUAL_DISTANCE":600,
     "PUPIL_SPEED_MAD_N":16,
     "PUPIL_SPEED_TH":None,
     "BLINK_PAD":0.05,
     "BLINK_PAD_MINLEN":0.075,
     "BLINK_PAD_ADAPTIVE_V":1.5,
     "BLINK_PAD_MAX_PRE":0.1,
     "BLINK_PAD_MAX_POST":0.3,
     "BLINK_MIN_ISLAND":0.1,
     "normFlag":False,
     "s_trg":"Start_Experiment",
     "visualization":False,
     "MS":False,
     "rejectFlag":[]
     }


def run(iSub, cfg):

    cfg = cfg.copy()

    
    mmName=[
        "PDR",
        "zeroArray",
        "PDR_trial",
        "condition",
        "analysis",
        "t1",
        "t2",
        "res_t1",
        "res_t2",
        "RT_t1",
        "RT_t2",
        "lag_cond",
        "lag",
        "sub",
        "run",
        ]
    
    datHashRun={}
    datHashRun = {name: [] for name in mmName}
    
    print("Processing --> " + iSub[-3:] + "...")
    ascFileName = sorted(glob.glob(f"{iSub}/**/*.asc"))
    

    for iRun,ascFile in enumerate(ascFileName):
        
        t2a = asc2array_cls(cfg) # make instance
        t2a.cfg["fName"]= f"preprocess/{iSub.split('/')[-1]}/run{iRun}_{dt_now.strftime('%Y%m%d')}"

        dat = t2a.dataExtraction(ascFile)
    
        eyeData,events,initialTimeVal,fs = t2a.dataParse(dat)


        eyeData = t2a.blinkInterp(eyeData)


        # better eye per run (cfg["usedEye"] = 1), as in Exp1/parseData.py

        if cfg["mmFlag"]:
            eyeData["pupilData"] = au2mm(eyeData["pupilData"], t2a.Method, eyeData["usedEye"], vis_dist=cfg["VISUAL_DISTANCE"])

        onset_trial = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "START_TRIAL" in e[1]][0]

        pupilData = eyeData["pupilData"].reshape(-1)


        tmp_pupilData=[]
        timeLen=2
        for ev in [int(e[0])-initialTimeVal for e in events["MSG"] if "TROUGH_DETECTED" in e[1]]:
            # aaaa
            if int(ev+timeLen*fs) > len(pupilData):
                tmp_zeropad=np.zeros(int(ev+timeLen*fs)-len(pupilData))
                tmp_pupilData.append(np.r_[pupilData[np.arange(int(ev-timeLen*fs),len(pupilData))],tmp_zeropad])
            else:
                tmp_pupilData.append(pupilData[np.arange(int(ev-timeLen*fs),int(ev+timeLen*fs))])
        
        datHashRun["PDR_trial"].append(np.array(tmp_pupilData).tolist())
            
        onset_trial = [int(e[0]) for e in events["MSG"] if "TRIAL" == e[1]]
        onset_trial.append(int(events["MSG"][-1][0]))
        
        res_correct={"t1":[],
                     "t2":[],
                     "t1_RT":[],
                     "t2_RT":[],
                     "lag":[]
                     }

        for o in np.arange(len(onset_trial)-1):

            t = [e for e in events["MSG"] if (int(e[0]) >= onset_trial[o]) & (int(e[0])<onset_trial[o+1])]
            cond = [e[2] for e in t if "Condition" in e[1]][0]
            
            t1_onset = [int(e[0]) for e in t if "ONSET_T1" in e[1]]    
            t2_onset = [int(e[0]) for e in t if "ONSET_T2" in e[1]]    
    
            res = [[int(e[0]),e[2]] for e in t if ("KEY_INPUT" in e[1])]
            res_left = [int(e[0]) for e in res if ("left" in e[1])]
            res_right = [int(e[0]) for e in res if ("right" in e[1])]

            if len(res_left)>0:
                res_correct["t1"].append(1)
                if len(t1_onset)>0:
                    res_correct["t1_RT"].append(res_left[0]-t1_onset[0])
                else:
                    res_correct["t1_RT"].append(-1)
            else:
                res_correct["t1"].append(0)
                res_correct["t1_RT"].append(-1)
                
            if len(res_right)>0:
                res_correct["t2"].append(1)
                if len(t2_onset)>0:
                    res_correct["t2_RT"].append(res_right[0]-t2_onset[0])
                else:
                    res_correct["t2_RT"].append(-1)
            else:
                res_correct["t2"].append(0)
                res_correct["t2_RT"].append(-1)
            
            if cond=="Both":
                res_correct["lag"].append(t2_onset[0]-t1_onset[0])
            else:
                res_correct["lag"].append(-1)
                
        datHashRun["PDR"].append(pupilData.tolist())
        # 1 = interpolated (blink / outlier) sample, same time axis as pupilData
        datHashRun["zeroArray"].append(np.asarray(eyeData["zeroArray"]).reshape(-1).astype(int).tolist())
        datHashRun["condition"].append([e[2] for e in events["MSG"] if "Condition" in e[1]])
        datHashRun["analysis"].append(ascFile.split('/')[-2])
        
        for t1t2 in [1,2]:
            datHashRun[f"t{t1t2}"].append([int(int(e[0])- initialTimeVal) for e in events["MSG"] if f"ONSET_T{t1t2}" in e[1]])
            datHashRun[f"RT_t{t1t2}"].append(res_correct[f"t{t1t2}_RT"])
            datHashRun[f"res_t{t1t2}"].append(res_correct[f"t{t1t2}"])

        datHashRun["lag_cond"].append([int(e[2]) for e in events["MSG"] if "Lag" in e[1]])
        datHashRun["lag"].append(res_correct["lag"])
        datHashRun["sub"].append(iSub.split('/')[-1])
        datHashRun["run"].append(iRun)

    cfg["SAMPLING_RATE"] = fs
    
    for mm in list(datHashRun.keys()):
        if not isinstance(datHashRun[mm],list):
            datHashRun[mm] = datHashRun[mm].tolist()
        
    for i in np.arange(len(datHashRun["PDR"])):
        nOut = int(len(datHashRun["PDR"][i])*(cfg["RESAMPLING_RATE"]/fs))
        datHashRun["PDR"][i] = re_sampling(np.array(datHashRun["PDR"][i]).reshape(1,-1),nOut,keep_nan=True)[0].tolist()

        z = np.array(datHashRun["zeroArray"][i])
        edges = (np.arange(nOut)*len(z)/nOut).astype(int)
        datHashRun["zeroArray"][i] = np.maximum.reduceat(z, edges).tolist()
    
    
    if not os.path.exists(f"{date}"):
        os.makedirs(f"{date}",exist_ok=True)
    
    with open(f"{date}/{iSub[-3:]}_run.msgpack", "wb") as f:
        f.write(msgpack.packb(datHashRun))
    
    return cfg

def speed_th(iSub, cfg):
    """pupil speed threshold [left, right] (mm/s) of one participant, all runs pooled"""
    return asc2array_cls(cfg.copy()).pupilSpeedThreshold(sorted(glob.glob(f"{iSub}/**/*.asc")))

if __name__ == '__main__':

    # per-participant pupil speed threshold, from all their runs
    speedTh = Parallel(n_jobs=8)(delayed(speed_th)(iSub, cfg) for iSub in folderName)
    speedTh = dict(zip(folderName, speedTh))

    tmp_cfg = Parallel(n_jobs=8)(delayed(run)(iSub, {**cfg, "PUPIL_SPEED_TH": speedTh[iSub]}) for iSub in folderName)

    tmp_cfg = [c for c in tmp_cfg if c is not None]
    if len(tmp_cfg) > 0:
        # PUPIL_SPEED_TH differs between participants: saved in pupil_speed_th.json
        with open(f"{date}/cfg.json","w") as f:
            json.dump({**tmp_cfg[0], "PUPIL_SPEED_TH": None},f)
        with open(f"{date}/pupil_speed_th.json","w") as f:
            json.dump({os.path.basename(k): v for k, v in speedTh.items()},f,indent=1)
