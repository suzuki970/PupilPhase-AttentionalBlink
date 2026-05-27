#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:58:40 2021

@author: yutasuzuki
"""

import numpy as np
import json
from asc2array_cls import asc2array_cls

import glob
import os
import math
from pre_processing import pre_processing,moving_avg,re_sampling
import matplotlib.pyplot as plt
from itertools import chain
import datetime
from multiprocessing import Pool
import msgpack
        
from band_pass_filter import lowpass_filter
from pre_processing_cls import getPCPDevents


folderName = sorted(glob.glob("./Experimental_scripts/results/*"))

dt_now = datetime.datetime.now()
date = dt_now.strftime("%Y%m%d")

rootFolder = "./Experimental_scripts/results"

subNames = glob.glob(rootFolder+"/T1-active/*")
subNames = subNames + glob.glob(rootFolder+"/T1-passive/*")
subNames.sort()
    
cfg={"THRES_DIFF":10,
     "WID_ANALYSIS":3,
     "WID_SEQUENCE":1.28+3.2,
     "WID_BP_ANALYSIS":5,
     # "useEye":1,
     "WID_FILTER":[],
     "RESAMPLING_RATE":200,
     # "mmFlag":False,
     # "normFlag":True,
     "usedEye":1,
     "mmFlag":True,
     "normFlag":False,
     "s_trg":[],
     "visualization":False,
     "MS":False,
     "rejectFlag":[]
     }

if cfg['mmFlag']:
    unitName = ""
else:
    unitName = "_norm" 

# t2a = asc2array_cls(cfg) # make instance

# rootFolder = "../Experimental_scripts/results/"+sessionName+"/"

# folderList=[]
# for filename in os.listdir(rootFolder):
#     if os.path.isdir(os.path.join(rootFolder, filename)): 
#         folderList.append(filename)
# folderList.sort()

#%%
# iSub = subNames[-1]
def run(iSub):

    if os.path.exists(f"./data/{date}/{os.path.basename(iSub)}_{iSub.split('/')[-2]}_trial{unitName}.msgpack"):
        return
    
    # %%

    datHashRun={
        "PDR":[],
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
    rejectFlag = []
    
    for iRun,(cFile,ascFile) in enumerate(zip(fileName,ascFileName)):

        #% ################## data load from ##################
        if not fileName:
            print("Empty! skipped...")
            continue
        
        f = open(cFile)
        condition = json.load(f)
        f.close()
       
        rt0 = [d for d in condition["res"]["RT"]]
        responses0 = [d for d in condition["res"]["ans"]]
        
        t2a = asc2array_cls(cfg)
        t2a.cfg["fName"]= f"preprocess/{iSub.split('/')[-1]}/run{iRun}_{dt_now.strftime('%Y%m%d')}"

        #% ################## load json file (eye data) ##################
        dat = t2a.dataExtraction(ascFile)
    
        #% ################## load json file (eye data) ##################
        eyeData,events,initialTimeVal,fs = t2a.dataParse(dat)
        # ave,sigma = t2a.getAve(eyeData["pupilData"])
        
        # pupil_bef=eyeData["pupilData_original"]
        #% ################## load json file (eye data) ##################
        eyeData = t2a.blinkInterp(eyeData)
        
        # pupil_aft = eyeData["pupilData"]
        # y_lowpassed = lowpass_filter(pupil_aft.reshape(1,-1), 0.15, fs)
        # PCPDevents = getPCPDevents(y_lowpassed,0.001)

        # x = np.arange(len(pupil_bef[1,:]))*(1/fs)
        
        # plt.plot(x,pupil_bef[1,:])
        # plt.plot(x,pupil_aft[0,:])
        # plt.plot(x,y_lowpassed[0,:])
        # plt.xlim([150,300])
        
        # trough = np.array(PCPDevents["troughs"][0])
        # plt.plot(x[trough],y_lowpassed[0,trough],'ro',alpha=0.5)
        # plt.plot(x[trough],[5.5]*len(trough),'ro',alpha=0.5)
        

        # if cfg["normFlag"]:
        #     pupilData = t2a.pupilNorm(eyeData["pupilData"], ave, sigma).reshape(-1)
        # else:
        pupilData = eyeData["pupilData"].reshape(-1)
            
        # pupilData = np.mean(pupilData,axis=0)
 
        gazeX = np.mean(eyeData["gazeX"],axis=0)
        gazeY = np.mean(eyeData["gazeY"],axis=0)
        
        rejectFlag.append(eyeData["rejectFlag"])
          
        #% ################## pre-processing ##################
        
        # eyeData,events,initialTimeVal,fs = asc2array(dat, cfg)
        
        # zeroInd = eyeData["zeroArray"].copy()

        # eyeName = ["R","L"]
        # tmp = []
        # for iEye in np.arange(len(eyeName)):
        #     zeroInd[iEye] = zeroInd[iEye]+initialTimeVal
        
        #     a = np.r_[0,np.argwhere(np.diff(zeroInd[iEye]) > 1).reshape(-1)]
            
        #     for iBlink in np.arange(len(a)-1):
        #         b = zeroInd[iEye][a[iBlink]+1:a[iBlink+1]]
                
        #         if (len(b) > 0.05*fs) & (len(b) < 2*fs):
        #             tmp.append([eyeName[iEye],b[0],b[-1]+1])
                    
        # tmp = sorted(tmp, key=lambda x:x[1])
        # events["EBLINK"] = tmp
        
        cfg["usedEye"] = eyeData["usedEye"]
        
        seq_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "seq" in e[1]]       
        
        # test = [[int(int(e[0])- initialTimeVal),(e[1])] for e in events["MSG"] if "seq" in e[1] or "T1" in e[1] or "Probe" in e[1] or "target" in e[1] or "probe" in e[1]]       
      
        t1_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "T1" in e[1]]       
        t2_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "Probe" in e[1]]       
        # t1t2_onset = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "T1" in e[1] or "Probe" in e[1]]          
     
        start_trial = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if e[1] == "Start_Pesentation"]
        end_trial = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if e[1] == "End_Pesentation"]
    
        events_res_target = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "target" in e[1]]    
        events_res_probe  = [[int(int(e[0])- initialTimeVal),e[1]] for e in events["MSG"] if "probe" in  e[1]]    
        
        condition_target = [[int(int(e[0])- initialTimeVal),int(e[1][-1])] for e in events["MSG"] if "Condition" in e[1]]    
        condition_lag    = [[int(int(e[0])- initialTimeVal),int(e[1][-1])] for e in events["MSG"] if "Lag" in e[1]]    
        
        if len(condition_lag)==0:
            condition_lag = [[c,c] for c in condition["condition_frame"]["t2_order"]]
        
        # ##################   ##################
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
        # events_t1_correct = [d for i,d in enumerate(events_t1_correct) if not i in rejectNum]
        # events_t2_correct = [d for i,d in enumerate(events_t2_correct) if not i in rejectNum]
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
            
        ################### RT ##################
        # events_t1_correct = []
        RT_t1 = []
        for iseq in np.arange(len(seq_onset)-1):
                
            t1_exist = False
            flg_multi = False
            for it1 in np.arange(len(t1_onset)-1):
                
                if t1_onset[it1][0] >= seq_onset[iseq][0] and t1_onset[it1][0] < seq_onset[iseq+1][0]: # if t1 exists between seq.
                    tmp = []
                    # print(str(iseq))
                    for ires_t1 in events_res_target:
                        # if ires_t1[0] >= t1_onset[it1][0] and ires_t1[0] < t1_onset[it1+1][0]:
                        if ires_t1[0] >= t1_onset[it1][0] and ires_t1[0] < t1_onset[it1][0]+(cfg["WID_SEQUENCE"]*fs):
                            tmp.append(1)
                            tmp2 = ires_t1[0]-t1_onset[it1][0]
                    
                    if len(tmp) > 0:
                        # events_t1_correct.append(1)
                        RT_t1.append(tmp2)
                    
                    else:
                        # events_t1_correct.append(0)
                        RT_t1.append(0)
                        
                    t1_exist = True  
                    flg_multi = True
                    
            if not t1_exist:
                # events_t1_correct.append(0)
                RT_t1.append(0)
        
        for ires_t1 in events_res_target:
            tmp = []
            if ires_t1[0] >= t1_onset[-1][0]:
                tmp.append(1)
                tmp2 = ires_t1[0]-t1_onset[-1][0]
                
        if len(tmp) > 0:
            # events_t1_correct.append(1)
            RT_t1.append(tmp2)
        else:
            # events_t1_correct.append(0)
            RT_t1.append(0)
        
        
        # events_t2_correct = []
        RT_t2 = []
        for iseq in  np.arange(len(seq_onset)-1):
            t2_exist = False
            for it2 in np.arange(len(t2_onset)-1):
                if t2_onset[it2][0] >= seq_onset[iseq][0] and t2_onset[it2][0] < seq_onset[iseq+1][0]:
                    tmp = []
                    for ires_t2 in events_res_probe:
                        if ires_t2[0] >= t2_onset[it2][0] and ires_t2[0] < t2_onset[it2][0]+(cfg["WID_SEQUENCE"]*fs):
                            tmp.append(1)
                            tmp2 = ires_t2[0]-t2_onset[it2][0]
                            
                    if len(tmp) > 0:
                        # events_t2_correct.append(1)
                        RT_t2.append(tmp2)
                    else:
                        # events_t2_correct.append(0)
                        RT_t2.append(0)
                   
                    t2_exist = True  
                    
            if not t2_exist:
                # events_t2_correct.append(0)
                RT_t2.append(0)
        
        for ires_t2 in events_res_probe:
            tmp = []
            if ires_t2[0] > t2_onset[-1][0]:
                tmp.append(1)
                tmp2 = ires_t2[0]-t2_onset[-1][0]
                
        if len(tmp) > 0:
            # events_t2_correct.append(1)
            RT_t2.append(tmp2)
        else:
            # events_t2_correct.append(0)
            RT_t2.append(0)
       
        responses = [[t1c,t2c] for t1c,t2c in zip(events_t1_correct,events_t2_correct)]
        rt = [[t1c,t2c] for t1c,t2c in zip(RT_t1,RT_t2)]
        

        # ################## eye metrics ##################
        timeLen = int(cfg["WID_BP_ANALYSIS"]*fs)
        initTime = seq_onset[0][0]
        for s in seq_onset:
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
        baselineWin = int(3*fs)
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
                    
                    if len(tmp) < (timeLen*2):
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
                    
                    if len(tmp) < (timeLen*2):
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
   
    cfg["rejectFlag"].append(rejectFlag)
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
        datHash[mmName] = re_sampling(datHash[mmName],int(len(datHash[mmName][0])*(cfg["RESAMPLING_RATE"]/fs))).tolist()
   
    for i in np.arange(len(datHashRun["PDR"])):
        datHashRun["PDR"][i] = re_sampling(np.array(datHashRun["PDR"][i]).reshape(1,-1),
                                           int(len(datHashRun["PDR"][i])*(cfg["RESAMPLING_RATE"]/fs))).tolist()

    if not os.path.exists("./data/"+date):
        os.mkdir("./data/"+date)

    with open(f"./data/{date}/{os.path.basename(iSub)}_{iSub.split('/')[-2]}_trial{unitName}.msgpack", "wb") as f:
        f.write(msgpack.packb(datHash))

    with open(f"./data/{date}/{os.path.basename(iSub)}_{iSub.split('/')[-2]}_run{unitName}.msgpack", "wb") as f:
        f.write(msgpack.packb(datHashRun))
    
    return cfg

if __name__ == '__main__':
 
    with Pool(10) as p:
        tmp_cfg = p.map(run, subNames)
    
    with open(f"./data/{date}/cfg.json","w") as f:
      json.dump(tmp_cfg[0],f)
 
