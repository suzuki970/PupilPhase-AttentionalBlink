#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:58:40 2021

@author: yutasuzuki
"""

# from asc2array import asc2array
# import math
# from itertools import chain
# from tqdm import tqdm

import msgpack,os,json,glob,datetime,random
import numpy as np
import pandas as pd
from multiprocessing import Pool

import matplotlib.pyplot as plt

from band_pass_filter import lowpass_filter
from pre_processing_cls import getPCPDevents
from pre_processing import pre_processing,moving_avg,re_sampling
from asc2array_cls import asc2array_cls

dt_now = datetime.datetime.now()
today = dt_now.strftime("%Y%m%d")
date = sorted(glob.glob("./data/*"))[-1]
date = f"./data/{today}"

folderName = sorted(glob.glob("../results/*"))
random.shuffle(folderName)

# %%

cfg={
     "RESAMPLING_RATE":200,
     # "usedEye":"Both",
     "usedEye":1,
     # "usedEye":'L',
     "WID_FILTER":[],
     "mmFlag":True,
     # "mmFlag":False,
     "normFlag":False,
     # "normFlag":False,
     "s_trg":"Start_Experiment",
     # "visualization":False,
     "visualization":True,
     "MS":False,
     "rejectFlag":[]
     }

#%%
# iSub=folderName[-1]
def run(iSub):
                   
    if os.path.exists(f"{date}/{iSub[-3:]}_run.msgpack"):
        print(f"{iSub[-3:]} skipped...")
        return
    
    mmName=[
        "PDR",
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
    # df=[]
    ascFileName = sorted(glob.glob(f"{iSub}/**/*.asc"))
    

    for iRun,ascFile in enumerate(ascFileName):
        
        t2a = asc2array_cls(cfg) # make instance
        t2a.cfg["fName"]= f"preprocess/{iSub.split('/')[-1]}/run{iRun}_{dt_now.strftime('%Y%m%d')}"

        dat = t2a.dataExtraction(ascFile)
    
        eyeData,events,initialTimeVal,fs = t2a.dataParse(dat)

        # if cfg["usedEye"]=="L":
        #     pupil_nointep = eyeData["pupilData_original"][0,:]
        # else:
        #     pupil_nointep = eyeData["pupilData_original"][1,:]
        # eyeData["pupilData"] = (np.pi * eyeData["pupilData"]**2) / 4

        eyeData = t2a.blinkInterp(eyeData)

        # p=eyeData["pupilData"].T
        # p[eyeData["zeroArray"]==1]=np.nan
        
        # x=np.arange(eyeData["pupilData"].shape[1])*(1/fs)
        # plt.plot(x,p)
        # plt.ylim(5,6)
        # plt.xlim(198,200)
                        
        if iRun == 0:
            cfg["usedEye"] = eyeData['betterEye']

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
        datHashRun["condition"].append([e[2] for e in events["MSG"] if "Condition" in e[1]])
        datHashRun["analysis"].append(ascFile.split('/')[-2])
        
        for t1t2 in [1,2]:
            datHashRun[f"t{t1t2}"].append([int(int(e[0])- initialTimeVal) for e in events["MSG"] if f"ONSET_T{t1t2}" in e[1]])
            datHashRun[f"RT_t{t1t2}"].append(res_correct[f"t{t1t2}_RT"])
            datHashRun[f"res_t{t1t2}"].append(res_correct[f"t{t1t2}"])

        datHashRun["lag_cond"].append([int(e[2]) for e in events["MSG"] if "Lag" in e[1]])
        datHashRun["lag"].append(res_correct["lag"])
        datHashRun["sub"].append(iSub.split('/')[-1])
        # datHashRun["run"].append(iRun if iRun < 3 else iRun-3)
        datHashRun["run"].append(iRun)

    cfg["SAMPLING_RATE"] = fs
    
    for mm in list(datHashRun.keys()):
        if not isinstance(datHashRun[mm],list):
            datHashRun[mm] = datHashRun[mm].tolist()
        
    for i in np.arange(len(datHashRun["PDR"])):
        datHashRun["PDR"][i] = re_sampling(np.array(datHashRun["PDR"][i]).reshape(1,-1),
                                           int(len(datHashRun["PDR"][i])*(cfg["RESAMPLING_RATE"]/fs)))[0].tolist()
    
    # with open(f"{date}/{iSub[-3:]}_trial.json","w") as f:
    #     json.dump(datHashRun,f)
    
    if not os.path.exists(f"{date}"):
        os.mkdir(f"{date}")
    
    with open(f"{date}/{iSub[-3:]}_run.msgpack", "wb") as f:
        f.write(msgpack.packb(datHashRun))
    
    with open(os.path.join(f"{date}/cfg.json"),"w") as f:
        json.dump(cfg,f)
    
#%%    
if __name__ == '__main__':
         
    with Pool(10) as p:
        tmp_cfg = p.map(run, folderName)

    # os.rename(date, f"./data/{today}")