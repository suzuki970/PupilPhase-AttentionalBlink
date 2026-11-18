#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import os
## run from this folder, also when executed from the workspace root (e.g. Positron)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath("Exp1" if os.path.isdir("Exp1") else ".")
os.chdir(HERE)
sys.path.append(os.path.join(HERE, "..", "lib"))
import datetime
import glob
import json
import os

import msgpack
import numpy as np
from asc2array_cls import asc2array_cls
from au2mm import au2mm
from joblib import Parallel, delayed
from pre_processing import re_sampling


dt_now = datetime.datetime.now()
date = dt_now.strftime("%Y%m%d")

rootFolder = "./rawData"

subNames = sorted(glob.glob(rootFolder+"/T1-attend/*"))
subNames = sorted(subNames + glob.glob(rootFolder+"/T1-ignore/*"))
    
cfg={
     "WID_PRE_EVENT":3,
     "WID_ANALYSIS":3,
     "WID_SEQUENCE":1.28+3.2,
     "WID_BP_ANALYSIS":5,
     "RESAMPLING_RATE":200,
     "VISUAL_DISTANCE":700,
     "PUPIL_SPEED_MAD_N":16,
     "PUPIL_SPEED_TH":None,
     "BLINK_PAD":0.05,
     "BLINK_PAD_MINLEN":0.075,
     "BLINK_PAD_ADAPTIVE_V":1.5,
     "BLINK_PAD_MAX_PRE":0.1,
     "BLINK_PAD_MAX_POST":0.3,
     "BLINK_MIN_ISLAND":0.1,
     "usedEye":1,
     "mmFlag":True,
     "s_trg":[],
     "visualization":False,
     }

def run(iSub, cfg):

    cfg = cfg.copy()


    datHashRun={
        "PDR":[],
        "zeroArray":[],
        "responses":[],
        "sub":[],
        "RT":[],
        "lag":[],
        "t1":[],
        "t2":[],
        "target":[],
        "condition_target":[],
        "run":[],
        "session":[],
        }
    
    datHash={"PDR_t1":[],
             "PDR_t2":[],
             "gazeX_t1":[],
             "gazeY_t1":[],
             "gazeX_t2":[],
             "gazeY_t2":[],
             "session":[],
             "baseline":[],
             "seq_onset":[],
             "responses":[],
             "time":[],
             "sub":[],
             "RT":[],
             "run":[],
             "lag":[],
             "target":[],
             "Blink":[],
             "Saccade":[]
              }
    
    print("Analyzing " + iSub + "...")
  
    # %% load json file
    fileName = sorted(glob.glob(iSub + "/*.json"))
    ascFileName = sorted(glob.glob(iSub + "/*.asc"))
        
    # %% load json file
    
    for iRun,(cFile,ascFile) in enumerate(zip(fileName,ascFileName)):

        #% ################## data load from ##################
        if not fileName:
            print("Empty! skipped...")
            continue
        
        f = open(cFile)
        condition = json.load(f)
        f.close()
       

        t2a = asc2array_cls(cfg)
        t2a.cfg["fName"]= f"preprocess/{iSub.split('/')[-1]}/run{iRun}_{dt_now.strftime('%Y%m%d')}"

        #% ################## load json file (eye data) ##################
        dat = t2a.dataExtraction(ascFile)

        #% ################## load json file (eye data) ##################
        eyeData,events,initialTimeVal,fs = t2a.dataParse(dat)
        
        # Kret & Sjak-Shie rejection + interpolation; same eyeData keys as blinkInterp()

        eyeData = t2a.blinkInterp(eyeData)

        if cfg["mmFlag"]:
            eyeData["pupilData"] = au2mm(eyeData["pupilData"], t2a.Method, eyeData["usedEye"], vis_dist=cfg["VISUAL_DISTANCE"])

        pupilData = eyeData["pupilData"].reshape(-1)

        gazeX = np.mean(eyeData["gazeX"],axis=0)
        gazeY = np.mean(eyeData["gazeY"],axis=0)
        
        
        #% ################## pre-processing ##################
        
        seq_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "seq" in e[1]]       
        
      
        t1_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "T1" in e[1]]       
        t2_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "Probe" in e[1]]       
     
    
        events_res_target = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "target" in e[1]]    
        events_res_probe  = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "probe" in  e[1]]    
        
        condition_target = [[int(int(e[0])- initialTimeVal),int(e[1][-1])] for e in events["MSG"] if "Condition" in e[1]]    
        condition_lag    = [[int(int(e[0])- initialTimeVal),int(e[1][-1])] for e in events["MSG"] if "Lag" in e[1]]    
        
        if len(condition_lag)==0:
            condition_lag = [[c,c] for c in condition["condition_frame"]["t2_order"]]
        
        rejectNum = []
        for itar in  np.arange(len(condition_target)-1):
            tmp = []
                    
            for it1,it2 in zip(t1_onset,t2_onset):
                if it1[0] >= condition_target[itar][0] and it1[0] < condition_target[itar+1][0]:
                    tmp.append(1)
                if it2[0] >= condition_target[itar][0] and it2[0] < condition_target[itar+1][0]:
                    tmp.append(1)
        
            if len(tmp) == 0:
                rejectNum.append(itar)
        
        condition_target = [d for i,d in enumerate(condition_target) if not i in rejectNum]
        condition_lag = [d for i,d in enumerate(condition_lag) if not i in rejectNum]
        seq_onset = [d for i,d in enumerate(seq_onset) if not i in rejectNum]
   
        # ################## answer correctness check ##################
        events_t1_correct = np.zeros(len(seq_onset))
        events_t2_correct = np.zeros(len(seq_onset))
        for iseq in np.arange(len(seq_onset)-1): # false alarm
            tmp = []
            for ires_t1 in events_res_target:
                if ires_t1[0] >= seq_onset[iseq][0] and ires_t1[0] < seq_onset[iseq+1][0]: # if response for t1 between seq.
                    tmp.append(1)
            if len(tmp) > 0:
                events_t1_correct[iseq] = 1
    
            tmp = []
            for ires_t2 in events_res_probe:
                if ires_t2[0] >= seq_onset[iseq][0] and ires_t2[0] < seq_onset[iseq+1][0]: # if response for t2 between seq.
                    tmp.append(1)
            if len(tmp) > 0:
                events_t2_correct[iseq] = 1
        
        tmp = []
        for ires_t1 in events_res_target:
            if ires_t1[0] >= seq_onset[-1][0]:
                tmp.append(1)
        if len(tmp) > 0:
            events_t1_correct[-1] = 1
        
        tmp = []
        for ires_t2 in events_res_probe:
            if ires_t2[0] >= seq_onset[-1][0]:
                tmp.append(1)
        if len(tmp) > 0:
            events_t2_correct[-1] = 1
            
        RT_t1 = []
        for iseq in np.arange(len(seq_onset)-1):
                
            t1_exist = False
            for it1 in np.arange(len(t1_onset)-1):
                
                if t1_onset[it1][0] >= seq_onset[iseq][0] and t1_onset[it1][0] < seq_onset[iseq+1][0]: # if t1 exists between seq.
                    tmp = []
                    for ires_t1 in events_res_target:
                        if ires_t1[0] >= t1_onset[it1][0] and ires_t1[0] < t1_onset[it1][0]+(cfg["WID_SEQUENCE"]*fs):
                            tmp.append(1)
                            rt = ires_t1[0]-t1_onset[it1][0]
                    
                    if len(tmp) > 0:
                        RT_t1.append(rt)
                    
                    else:
                        RT_t1.append(0)
                        
                    t1_exist = True  
                    
            if not t1_exist:
                RT_t1.append(0)
        
        for ires_t1 in events_res_target:
            tmp = []
            if ires_t1[0] >= t1_onset[-1][0]:
                tmp.append(1)
                rt = ires_t1[0]-t1_onset[-1][0]
                
        if len(tmp) > 0:
            RT_t1.append(rt)
        else:
            RT_t1.append(0)
        
        
        RT_t2 = []
        for iseq in  np.arange(len(seq_onset)-1):
            t2_exist = False
            for it2 in np.arange(len(t2_onset)-1):
                if t2_onset[it2][0] >= seq_onset[iseq][0] and t2_onset[it2][0] < seq_onset[iseq+1][0]:
                    tmp = []
                    for ires_t2 in events_res_probe:
                        if ires_t2[0] >= t2_onset[it2][0] and ires_t2[0] < t2_onset[it2][0]+(cfg["WID_SEQUENCE"]*fs):
                            tmp.append(1)
                            rt = ires_t2[0]-t2_onset[it2][0]
                            
                    if len(tmp) > 0:
                        RT_t2.append(rt)
                    else:
                        RT_t2.append(0)
                   
                    t2_exist = True  
                    
            if not t2_exist:
                RT_t2.append(0)
        
        for ires_t2 in events_res_probe:
            tmp = []
            if ires_t2[0] > t2_onset[-1][0]:
                tmp.append(1)
                rt = ires_t2[0]-t2_onset[-1][0]
                
        if len(tmp) > 0:
            RT_t2.append(rt)
        else:
            RT_t2.append(0)
       
        responses = [[t1c,t2c] for t1c,t2c in zip(events_t1_correct,events_t2_correct)]
        rt = [[t1c,t2c] for t1c,t2c in zip(RT_t1,RT_t2)]
        

        # ################## eye metrics ##################
        timeLen = int(cfg["WID_BP_ANALYSIS"]*fs)
        initTime = seq_onset[0][0]
        for s in seq_onset:
            datHash["seq_onset"].append(s[0])
            tmp = pupilData[s[0]-timeLen:s[0]]
            if len(tmp) < timeLen:
                datHash["baseline"].append(np.zeros((timeLen)).tolist())
                datHash["time"].append((s[0]-initTime)/fs)
            else:
                datHash["baseline"].append(tmp.tolist())
                datHash["time"].append((s[0]-initTime)/fs)
        
        count_t1 = 0
        count_t2 = 0
        timeLen = int(cfg["WID_ANALYSIS"]*fs)
        baselineWin = int(cfg["WID_PRE_EVENT"]*fs)
        for i,cond in enumerate(condition_target):
            if cond[1] == 1:
                for mmName in ["PDR_t1","PDR_t2","gazeX_t1","gazeY_t1","gazeX_t2","gazeY_t2"]:
                    datHash[mmName].append(np.zeros((timeLen+baselineWin)).tolist())
            
            else:
                if (cond[1] == 2) | (cond[1] == 4):
                    if cond[1] == 2:
                        for mmName in ["PDR_t2","gazeX_t2","gazeY_t2"]:
                            datHash[mmName].append(np.zeros((timeLen+baselineWin)).tolist())
             
                    tmp = pupilData[t1_onset[count_t1][0]-baselineWin:t1_onset[count_t1][0]+timeLen]
                    tmp_gx = gazeX[t1_onset[count_t1][0]-baselineWin:t1_onset[count_t1][0]+timeLen]
                    tmp_gy = gazeY[t1_onset[count_t1][0]-baselineWin:t1_onset[count_t1][0]+timeLen]
                    
                    if len(tmp) < (timeLen+baselineWin):
                        datHash["PDR_t1"].append(np.r_[tmp,np.zeros((timeLen+baselineWin)-len(tmp))].tolist())
                        datHash["gazeX_t1"].append(np.r_[tmp_gx,np.zeros((timeLen+baselineWin)-len(tmp_gx))].tolist())
                        datHash["gazeY_t1"].append(np.r_[tmp_gy,np.zeros((timeLen+baselineWin)-len(tmp_gy))].tolist())
                    else:
                        datHash["PDR_t1"].append(tmp.tolist())
                        datHash["gazeX_t1"].append(tmp_gx.tolist())
                        datHash["gazeY_t1"].append(tmp_gy.tolist())
                          
                    count_t1 += 1
                    
                if (cond[1] == 3) | (cond[1] == 4):    
                    if cond[1] == 3:
                        for mmName in ["PDR_t1","gazeX_t1","gazeY_t1"]:
                            datHash[mmName].append(np.zeros((timeLen+baselineWin)).tolist())
             
                    tmp = pupilData[t2_onset[count_t2][0]-baselineWin:t2_onset[count_t2][0]+timeLen]
                    tmp_gx = gazeX[t2_onset[count_t2][0]-baselineWin:t2_onset[count_t2][0]+timeLen]
                    tmp_gy = gazeY[t2_onset[count_t2][0]-baselineWin:t2_onset[count_t2][0]+timeLen]
                    
                    if len(tmp) < (timeLen+baselineWin):
                        datHash["PDR_t2"].append(np.r_[tmp,np.zeros((timeLen+baselineWin)-len(tmp))].tolist())
                        datHash["gazeX_t2"].append(np.r_[tmp_gx,np.zeros((timeLen+baselineWin)-len(tmp_gx))].tolist())
                        datHash["gazeY_t2"].append(np.r_[tmp_gy,np.zeros((timeLen+baselineWin)-len(tmp_gy))].tolist())
                    
                    else:
                        datHash["PDR_t2"].append(tmp.tolist())
                        datHash["gazeX_t2"].append(tmp_gx.tolist())
                        datHash["gazeY_t2"].append(tmp_gy.tolist())
                   
                    count_t2 += 1
                    
        
            datHash["run"].append(iRun)
            datHash["session"].append(iSub.split('/')[-2])
            datHash["sub"].append(iSub.split('/')[-1])
            
        datHashRun["PDR"].append(pupilData.tolist())
        # 1 = interpolated (blink / outlier) sample, same time axis as pupilData
        datHashRun["zeroArray"].append(np.asarray(eyeData["zeroArray"]).reshape(-1).astype(int).tolist())
        datHashRun["condition_target"].append(condition_target)
        datHashRun["t1"].append(t1_onset)
        datHashRun["t2"].append(t2_onset)
        datHashRun["RT"].append(rt)
        datHashRun["responses"].append(responses)
        datHashRun["target"].append(np.array(condition_target)[:,1].tolist())
        datHashRun["lag"].append(np.array(condition_lag)[:,1].tolist())
        datHashRun["sub"].append(iSub.split('/')[-1])
        datHashRun["run"].append(iRun)
        datHashRun["session"].append(iSub.split('/')[-2])

        # ################## for heatmap, blink and saccade ##################
        event_data = {"EFIX":[],"ESACC":[],"EBLINK":[]}
        for mm in list(event_data.keys()):
            for s in seq_onset:   
                tmp = []
                for e in events[mm]:
                    if (int(e[1])-initialTimeVal > (s[0]-baselineWin)) & (int(e[1])-initialTimeVal < (s[0]+timeLen)):
                        if e[0] == eyeData["usedEye"]:
                            tmp_e = e.copy()
                            tmp_e[1] = int(tmp_e[1])-initialTimeVal - s[0]
                            tmp_e[2] = int(tmp_e[2])-initialTimeVal - s[0]
                            tmp.append(tmp_e)
                   
                event_data[mm].append(tmp)
        
        for e in event_data["EBLINK"]:      
            datHash["Blink"].append([[e_data[1],e_data[2]] for e_data in e])
        for e in event_data["ESACC"]:      
            datHash["Saccade"].append([[e_data[2],e_data[8]] for e_data in e])
    
        # ################## data storing in dict ##################
        
        datHash["RT"]  = datHash["RT"]+ rt
        datHash["responses"]  = datHash["responses"]+ responses
        
        datHash["target"] = np.r_[datHash["target"],np.array(condition_target)[:,1]]
        datHash["lag"]  = np.r_[datHash["lag"], np.array(condition_lag)[:,1]]
   
    cfg["SAMPLING_RATE"] = fs


    #%% save files   
    
    for mmName in ["gazeX_t1","gazeY_t1","gazeX_t2","gazeY_t2"]:
        datHash[mmName] = np.array(datHash[mmName]).tolist()
    
    for mm in list(datHash.keys()):
        if not isinstance(datHash[mm],list):
            datHash[mm] = datHash[mm].tolist()

    for mm in list(datHashRun.keys()):
        if not isinstance(datHashRun[mm],list):
            datHashRun[mm] = datHashRun[mm].tolist()
        
    for mmName in ["PDR_t1","PDR_t2","baseline","gazeX_t1","gazeY_t1","gazeX_t2","gazeY_t2"]:
        # keep_nan: the Kret pipeline leaves gaps > 250 ms as NaN -- keep them
        datHash[mmName] = re_sampling(datHash[mmName],int(len(datHash[mmName][0])*(cfg["RESAMPLING_RATE"]/fs)),keep_nan=True).tolist()
   
    for i in np.arange(len(datHashRun["PDR"])):
        nOut = int(len(datHashRun["PDR"][i])*(cfg["RESAMPLING_RATE"]/fs))
        datHashRun["PDR"][i] = re_sampling(np.array(datHashRun["PDR"][i]).reshape(1,-1),nOut,keep_nan=True).tolist()

        z = np.array(datHashRun["zeroArray"][i])
        edges = (np.arange(nOut)*len(z)/nOut).astype(int)
        datHashRun["zeroArray"][i] = np.maximum.reduceat(z, edges).reshape(1,-1).tolist()

    if not os.path.exists("./data/"+date):
        os.mkdir("./data/"+date)

    with open(f"./data/{date}/{os.path.basename(iSub)}_{iSub.split('/')[-2]}_trial.msgpack", "wb") as f:
        f.write(msgpack.packb(datHash))

    with open(f"./data/{date}/{os.path.basename(iSub)}_{iSub.split('/')[-2]}_run.msgpack", "wb") as f:
        f.write(msgpack.packb(datHashRun))
    
    return cfg

def speed_th(sub, cfg):
    """pupil speed threshold [left, right] (mm/s) of one participant, all runs of both sessions pooled"""
    return asc2array_cls(cfg.copy()).pupilSpeedThreshold(sorted(glob.glob(f"{rootFolder}/*/{sub}/*.asc")))

if __name__ == '__main__':

    # per-participant pupil speed threshold, from all their runs
    subs = sorted({os.path.basename(s) for s in subNames})
    speedTh = Parallel(n_jobs=8)(delayed(speed_th)(sub, cfg) for sub in subs)
    speedTh = dict(zip(subs, speedTh))

    tmp_cfg = Parallel(n_jobs=8)(delayed(run)(iSub, {**cfg, "PUPIL_SPEED_TH": speedTh[os.path.basename(iSub)]}) for iSub in subNames)
    tmp_cfg = [c for c in tmp_cfg if c is not None]
    if len(tmp_cfg) > 0:
        # PUPIL_SPEED_TH differs between participants: saved in pupil_speed_th.json
        with open(f"./data/{date}/cfg.json","w") as f:
            json.dump({**tmp_cfg[0], "PUPIL_SPEED_TH": None},f)
        with open(f"./data/{date}/pupil_speed_th.json","w") as f:
            json.dump(speedTh,f,indent=1)


