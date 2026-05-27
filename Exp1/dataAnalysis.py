#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:08:37 2021

@author: yutasuzuki
"""

import numpy as np
import matplotlib.pyplot as plt
from pre_processing import split_list,split_list2
from pre_processing_cls import pre_processing,rejectDat,vertical_line,rejectedByOutlier,SDT,getNearestValue
from band_pass_filter import lowpass_filter
from rejectBlink_PCA import rejectBlink_PCA
import json
import os
# import random
import warnings
# from pixel_size import pixel2angle
# import itertools
# from zeroInterp import zeroInterp
# import scipy.stats as sp
import pandas as pd
from makeEyemetrics import makeMicroSaccade,draw_heatmap
import datetime
import glob
import seaborn as sns
# from tqdm import tqdm
import msgpack

sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")

warnings.simplefilter("ignore")

#%% ------------------ initial settings ------------------

cfg={
"windowL":[20],
"TIME_START":-3,
"TIME_END":3,
"WID_FILTER":np.array([]),
"METHOD":1, #subtraction
"FLAG_LOWPASS":False,
"WID_BASELINE":[[-0.2,0]],
"DOT_PITCH":0.33,   
"VISUAL_DISTANCE":60,
"acceptMSRange":2,
"SCREEN_RES":[640*2, 512*2],
"SCREEN_MM":[432, 324],
"visualization":False,
}

dt_now = datetime.datetime.now()
date = dt_now.strftime("%Y%m%d")

folderName = glob.glob(os.path.join("./data/*"))
folderName.sort()
folderName = folderName[-1]

f = open(folderName + "/cfg.json")
cfg.update(json.load(f))
f.close()

# cfg["SAMPLING_RATE"] = cfg["RESAMPLING_RATE"]
# cfg["windowL"] = [20]

# if not cfg["mmFlag"] and not cfg["normFlag"]:
#     unitName = "_au"
#     cfg["THRES_DIFF"] = 20
# elif cfg["mmFlag"]:
#     unitName = "mm"
#     cfg["THRES_DIFF"] = 20
# else:
#     unitName = "norm" 
#     cfg["THRES_DIFF"] = 100

reject = {}

pp = pre_processing(cfg)

#%% ------------------ data loading -----------------------------------------

print("data loding...")

datHash = {}

# log = glob.glob(folderName+"/*_trial.json")
# log = glob.glob(folderName+"/*_trial.json")
log = glob.glob(folderName+"/*_trial.msgpack")
log.sort()

for subName in log:
    # f = open(subName)
    # tmp_datHash = json.load(f)
    # f.close()
    with open(subName, "rb") as f:
        tmp_datHash = msgpack.unpackb(f.read(), raw=False)

    for mmName in list(tmp_datHash.keys()):
        if mmName != "cfg":
            if not mmName in list(datHash.keys()):
                datHash[mmName] = []
            datHash[mmName] = datHash[mmName] + tmp_datHash[mmName]
    
time_x = np.linspace(cfg["TIME_START"], cfg["TIME_END"],int((cfg["TIME_END"]-cfg["TIME_START"])*cfg["SAMPLING_RATE"]))


#%% ------------------ artifact rejection -----------------------------------
print("pre-processing...")
y,reject["PDR_t1"]  = pp.pre_processing(np.array(datHash["PDR_t1"]).copy())
y2,reject["PDR_t2"] = pp.pre_processing(np.array(datHash["PDR_t2"]).copy())

x = np.linspace(cfg["TIME_START"],cfg["TIME_END"],y.shape[1])

#% ------------------ Hippus ---------------------------------------------

# datHash["hippus"],cfg["FFT_FREQ"],datHash["hippus_peak"] = pp.getHippus(np.array(datHash["baseline"])[:,getNearestValue(x,-3):getNearestValue(x,0)],2)


#% ------------------ baseline ---------------------------------------------

# plt.plot(np.array(datHash["baseline"]).mean(axis=0))
# plt.savefig(".figure/img0.pdf")
y_bp = np.array(datHash["baseline"])
x = np.linspace(-cfg["WID_BP_ANALYSIS"], 0, y_bp.shape[1])

y_bp = y_bp[:,getNearestValue(x,-1):getNearestValue(x,0)]

#########  rejected BP by velocity
tmp = abs(np.diff(y_bp))
ind = np.argwhere(tmp > 0.03)
# ind = np.argwhere(tmp > cfg["THRES_DIFF"])
reject["BP_diff"] = np.unique(ind[:,0])

dat, y, y2, y_bp = rejectDat(datHash, reject["BP_diff"].tolist(),y,y2,y_bp)

# plt.plot(np.diff(y_bp).T)

#########  rejected BP by outliar

reject["BP"] = rejectedByOutlier(datHash, y_bp)

dat, y, y2, y_bp = rejectDat(dat,reject["BP"],y,y2,y_bp)

#% ------------------ PCA --------------------------------------------------
# pca,reject["PCA_t1"] = rejectBlink_PCA(y)
# pca,reject["PCA_t2"] = rejectBlink_PCA(y2)

# tmp_reject = np.unique(reject["PCA_t1"].tolist() + reject["PCA_t2"].tolist()).tolist()

# dat, y, y2 = rejectDat(dat,tmp_reject,y,y2)

###### BP
    
pca,reject["PCA_BP"] = rejectBlink_PCA(y_bp)

dat, y, y2, y_bp = rejectDat(dat,reject["PCA_BP"],y,y2,y_bp)
 

#%% ------------------ reject subject (due to N < 50%) ----------------------
# tmp_reject=[]
# numOftrials = []
# numOftrials_res = []
# NUM_TRIAL = 480
# for i,iSub in enumerate(np.unique(datHash["sub"])):
#     ind0 = np.argwhere((np.array(datHash["sub"])==iSub) & (np.array(datHash["target"])==2))
#     ind1 = np.argwhere((np.array(datHash["sub"])==iSub) & (np.array(datHash["target"])==3))
#     ind2 = np.argwhere((np.array(datHash["sub"])==iSub) & (np.array(datHash["target"])==4))
#     numOftrials.append(len(ind0)+len(ind1)+len(ind2))
#     if (len(ind0)+len(ind1)+len(ind2)) < NUM_TRIAL * 0.5:
#             tmp_reject.append(iSub)
                
# tmp_reject = np.unique(tmp_reject)
# print("# of trials = " + str(numOftrials))
# print("Subject = " + str(tmp_reject) + " rejected by N")
   
# reject["Sub"] = [i for i,d in enumerate(datHash["sub"]) if d in tmp_reject]
 
# dat, y, y2, y_bp = rejectDat(dat,reject["Sub"],y,y2,y_bp)

#%% ------------------ make data frame --------------------------------------
df = pd.DataFrame()
for mmName in ["sub","session","lag","run","time"]:
    df[mmName] = datHash[mmName]

#### Target 1:no sound, 2:T1 only, 3:T2 only, 4:Both
g = ["no_sound","T1_only","T2_only","Both"]
df["condition"] = [g[int(t-1)] for t in datHash["target"]]

df["BaselinePD"] = y_bp.mean(axis=1)
df["EventPD_t1"] = np.array(datHash["PDR_t1"])[:,getNearestValue(time_x,-1):].mean(axis=1)
df["EventPD_t2"] = np.array(datHash["PDR_t2"])[:,getNearestValue(time_x,-1):].mean(axis=1)

df["RT_t1"] = np.array(datHash["RT"])[:,0]
df["RT_t1"][df["RT_t1"] == 0] = -1
df["RT_t2"] = np.array(datHash["RT"])[:,1]
df["RT_t2"][df["RT_t2"] == 0] = -1

df["res_t1"] = np.array(datHash["responses"])[:,0]
df["res_t2"] = np.array(datHash["responses"])[:,1]

# initialization
for mm in ["hit","miss","fa","cr"]:
    for t in ["_t1","_t2"]:
        df[mm+t]  = 0

#### T1-attend
mask = (df["condition"].isin(["T1_only", "Both"]))&(df["session"]=="T1-active")
df.loc[mask, "hit_t1"]  = (df["res_t1"] == 1).astype(int) # hit if responsded
df.loc[mask, "miss_t1"] = (df["res_t1"] == 0).astype(int) # miss if wasn"t responded

mask = df["condition"].isin(["T2_only"])
df.loc[mask, "fa_t1"]  = (df["res_t1"] == 1).astype(int) # FA if responsded when T1 is absent
df.loc[mask, "cr_t1"] = (df["res_t1"] == 0).astype(int) # CR if wasn"t responsded when T1 is absent

mask = df["condition"].isin(["T2_only", "Both"])
df.loc[mask, "hit_t2"]  = (df["res_t2"] == 1).astype(int) # hit if responsded
df.loc[mask, "miss_t2"] = (df["res_t2"] == 0).astype(int) # miss if wasn"t responded

mask = df["condition"].isin(["T1_only"])
df.loc[mask, "fa_t2"]  = (df["res_t2"] == 1).astype(int) # FA if responsded when T1 is absent
df.loc[mask, "cr_t2"] = (df["res_t2"] == 0).astype(int) # CR if wasn"t responsded when T1 is absent
                            
df["P(T2|T1)"] = 0
df["P(T2|T1)"][(df["hit_t1"]==1) & (df["hit_t2"]==1)]=1

# df_roll = df.copy()
# interestOfLag = 5
# for iLag in np.arange(-interestOfLag,interestOfLag+1):
#     tmp = []
#     for iSub in np.unique(df["sub"]):
#         for iSession in np.unique(df["session"]):
#             a = np.ones(len(df[(df["sub"]==iSub)&(df["session"]==iSession)]))*-100
#             if iLag < 0:
#                 a[:iLag] = np.roll(df[(df["sub"]==iSub)&(df["session"]==iSession)]["BaselinePD"],iLag)[:iLag]
#             elif iLag > 0:
#                 a[iLag:] = np.roll(df[(df["sub"]==iSub)&(df["session"]==iSession)]["BaselinePD"],iLag)[iLag:]
#             else:
#                 a = df[(df["sub"]==iSub)&(df["session"]==iSession)]["BaselinePD"]
                
#             tmp = np.r_[tmp,a]

#     df_roll["Lag"+str(iLag)] = tmp
    
# for iLag in np.arange(-interestOfLag,interestOfLag+1):
#     df_roll = df_roll[df_roll["Lag"+str(iLag)] != -100]

#%% reject sub

tmp_df = df[(df["session"]=="T1-active")]
tmp_df = tmp_df.groupby(["sub","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)

rejectSub=[]
for (c,h) in zip(["T1_only","T2_only","T2_only","T1_only"],["hit_t1","hit_t2","fa_t1","fa_t2"]):
# for (c,h) in zip(["T1_only"],["hit_t1"]):
    
    tmp_tmp_df = tmp_df[(tmp_df["condition"]==c)&(df["session"]=="T1-active")]

    z = zscore(tmp_tmp_df[h])
    mask = np.abs(z) > 3
        
    rejectSub.append(tmp_tmp_df[mask]["sub"].values)

    # sigma = np.std(tmp_tmp_df[h])
    
    # lower = np.mean(tmp_tmp_df[h]) - sigma*3
    # upper = np.mean(tmp_tmp_df[h]) + sigma*3
    
    # q75, q25 = np.percentile(tmp_tmp_df[h], [75 ,25])
    # IQR = q75 - q25

    # lower = q25 - IQR*4
    # upper = q75 + IQR*4

    # rejectSub.append(tmp_tmp_df[(tmp_tmp_df[h]<lower)|(tmp_tmp_df[h]>upper)]["sub"].values)

rejectSub = np.concatenate(rejectSub)
rejectSub = sorted(set(rejectSub))

print(f"{rejectSub} was rejected because of low accuracy for T1 or T2")

# for iSub in rejectSub:
#     df = df[df["sub"]!=iSub]
#     df_sum = df_sum[df_sum["sub"]!=iSub]
    
# reject = [i for i,d in enumerate(datHash["sub"]) if d in rejectSub]

# datHash = rejectDat(datHash,reject)

#%% ------------------ [Plot,reject] Accuracy for T1  -----------------------

plt.figure(figsize=(7,7))
plt.suptitle("T1", fontsize=20)

df_tmp = df[(df["condition"].isin(["T1_only", "Both"]))&
            (df["session"]=="T1-active")]
df_tmp = df_tmp.groupby(["sub"], sort=False, as_index=False).agg('mean', numeric_only=True)

plt.figure(figsize=(7,7))

sns.boxplot(y="hit_t1", data=df_tmp, showfliers=False)
sns.stripplot(y="hit_t1", data=df_tmp, jitter=True, color="black")
plt.ylabel("T1")
plt.ylim(0, 1)

df_tmp = df_tmp.reset_index()

reject["T1"] = list(df_tmp[df_tmp["hit_t1"] < 0.8]["sub"].values)

# print("# of trials = " + str(numOftrials))
print("Subject = " + str(reject["T1"]) + " rejected by T1 accuracy")

# for rej in reject["T1"]:
#     df = df[df["sub"] != rej]

# reject["T1"] = [i for i,d in enumerate(datHash["sub"]) if d in reject["T1"]]

# dat, y, y2, y_bp = rejectDat(dat,reject["T1"],y,y2,y_bp)


#%% ------------------ [Plot,reject] FA for T2 (only T1 presented) ----------
plt.figure(figsize=(7,7))
plt.suptitle("FA", fontsize=20)

df_tmp = df[(df["condition"].isin(["T1_only"]))&
            (df["session"]=="T1-active")]

df_tmp = df_tmp.groupby(["sub"], sort=False, as_index=False).agg('mean', numeric_only=True)

plt.figure(figsize=(7,7))

sns.boxplot(y="fa_t2", data=df_tmp, showfliers=False)
sns.stripplot(y="fa_t2", data=df_tmp, jitter=True, color="black")
plt.ylabel("False alarm(T2)")
plt.ylim(0, 1)

df_tmp = df_tmp.reset_index()

reject["FA"] = df_tmp[df_tmp["fa_t2"] > 0.8]["sub"].values

print("Subject = " + str(reject["FA"] ) + " rejected by FA")

# for rej in reject["FA"]:
#     df = df[df["sub"] != rej]

# reject["FA"] = [i for i,d in enumerate(datHash["sub"]) if d in reject["FA"]]

# dat, y, y2, y_bp = rejectDat(dat,reject["FA"],y,y2,y_bp)

#%% ------------------ MS ---------------------------------------------------
ev_t1,fs = makeMicroSaccade(cfg,datHash["gazeX_t1"],datHash["gazeY_t1"])
ev_t2,fs = makeMicroSaccade(cfg,datHash["gazeX_t2"],datHash["gazeY_t2"])

del datHash["gazeX_t1"], datHash["gazeY_t1"]
del datHash["gazeX_t2"], datHash["gazeY_t2"]

#%% ------------------ Blink and saccade ------------------------------------

rate,dat_MS_t1,datHash["sTimeOfMS_t1"] = pp.getBlink(datHash["Blink"], ev_t1)
rate2,dat_MS_t2,datHash["sTimeOfMS_t2"] = pp.getBlink(datHash["Blink"], ev_t2, 0.08*1000*np.array(datHash["lag"]))

datHash["BlinkRate"] = rate

del datHash["Blink"], datHash["Saccade"]

df_blink = pd.DataFrame()
for iSub in np.unique(df["sub"]):
    for iSession in np.unique(df["session"]):
        ind = np.argwhere((df["sub"].values==iSub) &
                          (df["session"].values==iSession)).reshape(-1)
        
        tmp_blink = pd.DataFrame()
        tmp_blink["rate"] = np.nanmean(np.array(rate)[ind,],axis=0)
        tmp_blink["sub"] = iSub
        tmp_blink["session"] = iSession
        tmp_blink["x"] = np.linspace(-cfg["WID_ANALYSIS"],cfg["WID_ANALYSIS"],len(rate[0]))

        df_blink = pd.concat([df_blink, tmp_blink])

df_tmp = df_blink.groupby(["x","session"], sort=False, as_index=False).agg('mean', numeric_only=True)
df_tmp = df_tmp.reset_index()
          
plt.figure()
plt.suptitle("Blink", fontsize=20)
sns.lineplot(x="x", y="rate", hue = "session", data=df_tmp)
plt.xlim([-1,3])
  
#%% ------------------ [Plot] number of trials ------------------------------
df["numOfTrials"] = 0

#### number of trials
for iSub in np.unique(df["sub"]):
    for iTarget in np.unique(df["target"]):
        for iSession in np.unique(df["session"]):
            for iLag in np.unique(df["lag"]):
                tmp =  df[(df["sub" ]== iSub) &
                          (df["target"] == iTarget) &
                          (df["lag"] == iLag) &
                          (df["session"]==iSession)
                          ]
                
                df["numOfTrials"][(df["sub" ]== iSub) &
                    (df["target"] == iTarget) &
                    (df["lag"] == iLag) &
                    (df["session"]==iSession)
                    ] = len(tmp)

#### P(T2|T1) for 
# for iSub in np.unique(df["sub"]):
#     for iTarget in np.unique(df["target"]):
#         for iSession in np.unique(df["session"]):
#             for iLag in np.unique(df["lag"]):
#                 if iTarget == 4:
#                     df_t1 = df[(df["sub"]==iSub) &
#                                (df["target"] == iTarget) &
#                                (df["lag"] == iLag) &
#                                (df["res_t1"]==1) & 
#                                (df["session"]==iSession)
#                               ]
#                 else:
#                     df_t1 = df[(df["sub"]==iSub) &
#                                (df["target"] == iTarget) &
#                                (df["lag"] == iLag) &
#                                (df["session"]==iSession)
#                               ]
                    
#                 df_t1_mean = df_t1.groupby(["session"]).agg(np.sum)
                
#                 df["P(T2|T1)"][(df["sub"]==iSub) &
#                                (df["target"] == iTarget) &
#                                (df["lag"] == iLag) &
#                                (df["session"]==iSession)
#                                ] = float(df_t1_mean["res_t2"].values / df_t1["numOfTrials"][:1].values)
                    
    # print("-----------------------------------")
    # print("Subject No." + str(iSub) + "...")
    # for iSession in np.unique(df["session"]):
        # print("---")
        # print("Session No." + str(iSession) + "...")
        # for iTarget in np.unique(df["target"]):
        #     print("Target No." + str(iTarget) + " = " + str(len(df[(df["session"]==iSession) & (df["target"]==iTarget) & (df["sub"]==iSub)])))
   
#%% ------------------ Tertile ----------------------------------------------

df_tertile_origin = df[(df["target"] == 4)].copy()
df_tertile = pd.DataFrame()

bp_tag = ["Small","Large"]


for iSub in np.unique(df_tertile_origin["sub"]):
    for iLag in np.unique(df_tertile_origin["lag"]):
        for iSession in np.unique(df_tertile_origin["session"]):
            # for iRun in np.unique(df_tertile_origin[df_tertile_origin["session"]==iSession]["run"]):
            tmp = df_tertile_origin[(df_tertile_origin["sub"] == iSub) & 
                                    (df_tertile_origin["lag"] == iLag) &
                                    (df_tertile_origin["session"] == iSession)
                                    # (df_tertile_origin["run"] == iRun)
                                    ]
            tmp = tmp.sort_values("BaselinePD")
           
            bp = tmp["BaselinePD"].values
           
            for iTertile,ter in enumerate(list(split_list2(bp,2))):
                t = tmp[ter[0][0]:ter[0][-1]+1]
                t["pupilSizeClass"] = bp_tag[iTertile]
                df_tertile = pd.concat([df_tertile, t])     

df_tmp = df_tertile.groupby(["sub","lag","session","pupilSizeClass"]).agg(np.mean)
df_tmp = df_tmp.groupby(["sub","pupilSizeClass"]).agg(np.mean)
df_tmp = df_tmp.reset_index()

fig = plt.figure()
sns.boxplot(x="pupilSizeClass", y="time", data=df_tmp, showfliers=False)
sns.stripplot(x="pupilSizeClass", y="time", data=df_tmp, jitter=True, color="black")


# for iSub in np.unique(df_tertile["sub"]):
#     for i,iSession in enumerate(np.unique(df_tertile_origin["session"])):
#         for iLag in np.unique(df_tertile["lag"]):
#             for iTag in np.unique(df_tertile["pupilSizeClass"]):
#                 df_t1 = df_tertile[(df_tertile["sub"]==iSub) &
#                                    (df_tertile["pupilSizeClass"] == iTag) &
#                                    (df_tertile["lag"] == iLag) &
#                                    (df_tertile["res_t1"]==1) & 
#                                    (df_tertile["session"] == iSession)
#                                    ]
                    
#                 df_t1_mean = df_t1.groupby(["session"]).agg(np.sum)
#                 df_tertile["P(T2|T1)"][(df_tertile["sub"]==iSub) &
#                                        (df_tertile["pupilSizeClass"] == iTag) &
#                                        (df_tertile["lag"] == iLag) &
#                                        (df_tertile["session"] == iSession)
#                                        # ((df_tertile["session"]==iSession[0]) | (df_tertile["session"]==iSession[1]))
#                                        ] = float(df_t1_mean["res_t2"].values / ((df_t1["numOfTrials"][:1].values)/(2*len(np.unique(df_tertile["pupilSizeClass"])))))

# df_tertile_origin = df
# df_tertileAll = pd.DataFrame()
   
# for iSub in np.unique(df_tertile_origin["sub"]):
#     for iLag in np.unique(df_tertile_origin["lag"]):
#         for iSession in np.unique(df_tertile_origin["session"]):
#         # for iSession in [[1,3],[2,4]]:
#             tmp = df_tertile_origin[(df_tertile_origin["sub"] == iSub) & 
#                                     (df_tertile_origin["lag"] == iLag) &
#                                     (df_tertile["session"] == iSession)
#                                     # ((df_tertile_origin["session"] == iSession[0]) |  (df_tertile_origin["session"] == iSession[1]))
#                                     ]
#             tmp = tmp.sort_values("BaselinePD")
           
#             bp = tmp["BaselinePD"].values
           
#             for iTertile,ter in enumerate(list(split_list2(bp,2))):
#                 t = tmp[ter[0][0]:ter[0][-1]+1]
#                 t["pupilSizeClass"] = iTertile
#                 df_tertileAll = pd.concat([df_tertileAll, t])     

#%% ------------------ SDT  -----------------------

# std_val = pd.DataFrame()
# for iSub in np.unique(df["sub"]):
#     for iSession in np.unique(df["session"]):
#         for iRun in np.unique(df["run"]):
#             tmp_sdt = df[(df["sub"]==iSub) &
#                          (df["session"]==iSession) &
#                          (df["run"]==iRun) &
#                          (df["target"]==4) &
#                          (df["res_t1"]==1)]
            
#             if len(tmp_sdt) > 0:
#                 # for i in np.arange(0,len(tmp_sdt)-60,10):
#                 tmp_sdt2 = tmp_sdt
#                 # [i:i+60]
#                 tmp_sdt2 = tmp_sdt2.groupby(["sub"]).agg(np.sum)
        
#                 tmp_std_val = SDT(tmp_sdt2["hit_t2"].values,tmp_sdt2["miss_t2"].values,tmp_sdt2["fa_t2"].values,tmp_sdt2["cr_t2"].values)
#                 tmp_std_val["sub"] = iSub
#                 tmp_std_val["session"] = iSession
#                 # tmp_std_val["roll"] = i
#                 tmp_std_val["run"] = iRun
#                 tmp_std_val["lag"] = tmp_sdt2["lag"].values
#                 std_val = pd.concat([std_val,tmp_std_val])

std_val = pd.DataFrame()
for iSub in np.unique(df_tertile["sub"]):
    for iSession in np.unique(df_tertile["session"]):
        # for iRun in np.unique(df_tertile["run"]):
        for iTag in np.unique(df_tertile["pupilSizeClass"]):
            tmp_sdt = df_tertile[(df_tertile["sub"]==iSub) &
                         (df_tertile["session"]==iSession) &
                         # (df_tertile["run"]==iRun) & 
                          (df_tertile["pupilSizeClass"]==iTag) &
                         (df_tertile["target"]==4) &
                         (df_tertile["res_t1"]==1)]
            
            if len(tmp_sdt) > 0:
                # for i in np.arange(0,len(tmp_sdt)-60,10):
                tmp_sdt2 = tmp_sdt
                # [i:i+60]
                tmp_sdt2 = tmp_sdt2.groupby(["sub"]).agg(np.sum)
        
                tmp_std_val = pd.DataFrame()
                tmp_std_val["y"] = tmp_sdt2["hit_t2"].values / (tmp_sdt2["hit_t2"].values+tmp_sdt2["miss_t2"].values)
                # SDT(tmp_sdt2["hit_t2"].values,tmp_sdt2["miss_t2"].values,tmp_sdt2["fa_t2"].values,tmp_sdt2["cr_t2"].values)
                tmp_std_val["sub"] = iSub
                tmp_std_val["session"] = iSession
                # tmp_std_val["roll"] = i
                # tmp_std_val["run"] = iRun
                tmp_std_val["pupilSizeClass"] = iTag
                tmp_std_val["lag"] = tmp_sdt2["lag"].values
                std_val = pd.concat([std_val,tmp_std_val])
    
# plt.figure()
# plt.suptitle("Blink", fontsize=20)
# sns.lineplot(x="roll", y="d", hue = "session", data=std_val)
          
# grid = sns.FacetGrid(std_val, col="session", row="sub", hue="session", size=5)
# grid.map(sns.pointplot, "roll", "d", dodge=True)
  
# grid = sns.FacetGrid(std_val, col="session", row="sub", hue="session", size=5)
# grid.map(sns.pointplot, "roll", "lag", dodge=True)
  
plt.figure()
# sns.boxplot(x="roll", y="d",hue="session", data=std_val, showfliers=False)
sns.boxplot(x="session", y="y",hue="pupilSizeClass", data=std_val, showfliers=False)

# plt.figure()
# sns.boxplot(x="roll", y="lag",hue="session", data=std_val, showfliers=False)
#%% ------------------ Tertile BP -------------------------------------------
# df_tertile_origin = df[(df["target"] == 4)].copy()
# df_tertile_BP = pd.DataFrame()
  
# for iSub in np.unique(df_tertile_origin["sub"]):
#     for iLag in np.unique(df_tertile_origin["lag"]):
#         for iSession in np.unique(df_tertile_origin["session"]):
#             tmp = df_tertile_origin[(df_tertile_origin["sub"] == iSub) & 
#                                     (df_tertile_origin["lag"] == iLag) &
#                                     (df_tertile_origin["session"] == iSession)
#                                     ]
#             tmp = tmp.sort_values("BaselinePD")
           
#             bp = tmp["BaselinePD"].values
           
#             for iTertile,ter in enumerate(list(split_list2(bp,3))):
#                 t = tmp[ter[0][0]:ter[0][-1]+1]
#                 t["tag_BP"] = iTertile
#                 df_tertile_BP = pd.concat([df_tertile_BP, t])     



# for iSub in np.unique(df_tertile_BP["sub"]):
#     # for i,iSession in enumerate([[1,3],[2,4]]):
#     for i,iSession in enumerate(np.unique(df_tertile_origin["session"])):
#         for iLag in np.unique(df_tertile_BP["lag"]):
#             for iTag in np.unique(df_tertile_BP["tag_BP"]):
#                 df_t1 = df_tertile_BP[(df_tertile_BP["sub"]==iSub) &
#                                    (df_tertile_BP["tag_BP"] == iTag) &
#                                    (df_tertile_BP["lag"] == iLag) &
#                                    (df_tertile_BP["res_t1"]==1) & 
#                                    (df_tertile_BP["session"] == iSession)
#                                    # ((df_tertile_BP["session"]==iSession[0]) | (df_tertile_BP["session"]==iSession[1]))
#                                    ]
                    
#                 df_t1_mean = df_t1.groupby(["session"]).agg(np.sum)
#                 df_tertile_BP["P(T2|T1)"][(df_tertile_BP["sub"]==iSub) &
#                                        (df_tertile_BP["tag_BP"] == iTag) &
#                                        (df_tertile_BP["lag"] == iLag) &
#                                        (df_tertile_BP["session"] == iSession)
#                                        # ((df_tertile_BP["session"]==iSession[0]) | (df_tertile_BP["session"]==iSession[1]))
#                                        ] = float(df_t1_mean["res_t2"].values / ((df_t1["numOfTrials"][:1].values)/(2*len(np.unique(df_tertile_BP["tag_BP"])))))

#%% ------------------ [Plot] Tertile BP ------------------------------------

# df_tmp = df_tertile.groupby(["sub","pupilSizeClass","lag","session"], sort=False, as_index=False).agg('mean', numeric_only=True)
# df_tmp = df_tmp.reset_index()

# # fig = plt.figure()
# grid = sns.FacetGrid(df_tmp, col="sub", hue="pupilSizeClass", col_wrap=5, size=5)
# grid.map(sns.pointplot, "lag", "P(T2|T1)", dodge=True)

# # ------------------ average ------------------
# plt.figure(figsize=(7,7))
# plt.title("Tertile", fontsize=20)

# df_tmp = df_tertile.groupby(["sub","pupilSizeClass","lag"], sort=False, as_index=False).agg('mean', numeric_only=True)
# df_tmp = df_tmp.reset_index()

# sns.pointplot(x="lag", y="P(T2|T1)", hue = "pupilSizeClass", data=df_tmp, dodge=True)
# plt.ylabel("Accuracy(P(T2))")
# plt.xlabel("Lag")
# plt.ylim(0, 1)

# plt.figure()
# plt.figure(figsize=(7,7))
# plt.suptitle("Tertile", fontsize=20)
# for iSub in np.unique(df_tertile["sub"]):
#     for iTag in np.unique(df_tertile["pupilSizeClass"]):
#         df_t1 = df_tertile[(df_tertile["sub"]==iSub) &
#                    (df_tertile["pupilSizeClass"]==iTag) &
#                    (df_tertile["session"]=="Attend")
#                 ]

#         df_t1 = df_t1.groupby(["session","lag"], sort=False, as_index=False).agg('mean', numeric_only=True)
    
#         plt.subplot(5,5,iSub)
#         if iTag == 0:
#             plt.plot(["1","2","3","4","5"],df_t1.loc[("Attend",slice(None)),"BaselinePD"],"b^")
#         else:
#             plt.plot(["1","2","3","4","5"],df_t1.loc[("Attend",slice(None)),"BaselinePD"],"ro")
            
#         plt.ylabel("BaselinePD")
#         plt.xlabel("Lag")
        
#%% ------------------ [Plot] Tertile P(T2|T1) ------------------------------

df_tmp = df_tertile[df_tertile["res_t1"] == 1]
df_tmp = df_tmp.groupby(["sub","lag","session","pupilSizeClass"], sort=False, as_index=False).agg('mean', numeric_only=True)
df_tmp = df_tmp.reset_index()

# fig = plt.figure()
# grid = sns.FacetGrid(df_tmp, col="sub", hue="pupilSizeClass", col_wrap=5, size=5)
# grid.map(sns.pointplot, "lag", "res_t2", dodge=True)

# ------------------ average ------------------
plt.figure(figsize=(7,7))

# grid = sns.FacetGrid(df_tmp, col="session", hue="pupilSizeClass", col_wrap=5, size=5)
grid = sns.FacetGrid(df_tmp, col="session", hue="pupilSizeClass", col_wrap=5)
grid.map(sns.pointplot, "lag", "res_t2", dodge=True)
plt.ylabel("Accuracy(P(T2))")
plt.xlabel("Lag")
# plt.ylim(0, 1)
        
#%% ------------------ [Plot] Accuracy for T2 only  -------------------------

df_tmp = df[df["target"]==3]
df_tmp = df_tmp.groupby(["sub"], sort=False, as_index=False).agg('mean', numeric_only=True)

plt.figure(figsize=(7,7))

sns.boxplot(x="target", y="hit_t2", data=df_tmp, showfliers=False)
sns.stripplot(x="target", y="hit_t2", data=df_tmp, jitter=True, color="black")
plt.ylabel("T2 only")
plt.xlabel("Lag")
plt.ylim(0, 1)
  
#%% ------------------ [Plot] Accuracy for T2 (P(T2|T1)) --------------------

df_tmp = df[(df["target"]==4) &
            (df["hit_t1"]==1) &
            (df["session"]=="Attend")]

df_tmp["trial"] = 0
for iSub in np.unique(df_tmp["sub"]):
    for iRun in np.unique(df_tmp["run"]):
        df_tmp_tmp = df_tmp[(df_tmp["sub"]==iSub) &
                            (df_tmp["run"]==iRun)]
        
        df_tmp_tmp["trial"] = np.arange(len(df_tmp_tmp))
        
        df_tmp[(df_tmp["sub"]==iSub) &
               (df_tmp["run"]==iRun)] = df_tmp_tmp
        
df_tmp = df_tmp[df_tmp["trial"] > 30]    

df_tmp["mvAve"] = 0
df_tmp["mvAvePupil"] = 0

st = 20
for iSub in np.unique(df_tmp["sub"]):
    for iRun in np.unique(df_tmp["run"]):
        df_tmp_tmp = df_tmp[(df_tmp["sub"]==iSub) &
                            (df_tmp["run"]==iRun)]
        
        for iStep in np.arange(0,len(df_tmp_tmp),int(st)):
            if iStep+st > len(df_tmp_tmp):
                df_tmp_tmp["mvAve"][iStep:iStep+1] = df_tmp_tmp[iStep:]["P(T2|T1)"].mean()
                df_tmp_tmp["mvAvePupil"][iStep:iStep+1] = df_tmp_tmp[iStep:]["BaselinePD"].mean()
            else:
                df_tmp_tmp["mvAve"][iStep:iStep+1] = df_tmp_tmp[iStep:(iStep+st)]["P(T2|T1)"].mean()
                df_tmp_tmp["mvAvePupil"][iStep:iStep+1] = df_tmp_tmp[iStep:(iStep+st)]["BaselinePD"].mean()
    
        df_tmp[(df_tmp["sub"]==iSub) &
               (df_tmp["run"]==iRun)] = df_tmp_tmp


df_tmp = df_tmp[df_tmp["mvAve"]!=0]

sns.pointplot(x="trial", y="mvAve", data=df_tmp, hue="run", ci="sd",dodge=True)
sns.pointplot(x="trial", y="mvAvePupil", data=df_tmp, hue="run", ci="sd",dodge=True)


fig = plt.figure(figsize=(10,7))
ax1 = fig.subplots()
ax2 = ax1.twinx()
ax2.grid(False)

sns.pointplot(x="trial", y="mvAve", data=df_tmp, ax=ax1, color="#D97168", ci="sd")
sns.pointplot(x="trial", y="mvAvePupil", data=df_tmp, ax=ax2, color="#738CD9",ci="sd")

df_tmp = df_tmp.groupby(["sub","session","lag"], sort=False, as_index=False).agg('mean', numeric_only=True)

df_tmp = df_tmp.reset_index()

# fig = plt.figure()
# grid = sns.FacetGrid(df_tmp, col="sub", hue="session", col_wrap=5, size=5)
# grid.map(sns.pointplot, "lag", "P(T2|T1)", dodge=True)
# plt.ylabel("Accuracy(P(T2))")
# plt.xlabel("Lag")
# plt.ylim(0, 1)
     
# ------------------ Average ------------------------------------

# plt.figure(figsize=(7,7))
# plt.suptitle("Averaged P(T2|T1)", fontsize=20)
# sns.pointplot(x="lag", y="hit_t2", hue = "session", data=df_tmp, dodge=True)
# plt.ylabel("Accuracy(P(T2))")
# plt.xlabel("Lag")
# plt.ylim(0, 1)

#%% ------------------ [Plot ] time-course ---------------------------------------------------
# df_roll_ave = df_roll
# df_roll_ave = df_roll_ave[(df_roll_ave["res_t1"]==1) & (df_roll_ave["target"]==4)]
# df_roll_ave = df_roll_ave.groupby(["session","res_t2"], sort=False, as_index=False).agg('mean', numeric_only=True)
# plt.plot(["1","2","3","4","5"],df_t1.loc[(0,slice(None)),"P(T2|T1)"],col[i])
        
    
#%% ------------------ save MS, tertile, compilation data -------------------

# path = os.path.join(folderName + "/data_tertile_"+unitName+".json")
# df_tertile.to_json(path)
# # path = os.path.join(folderName + "/data_tertile_all_"+unitName+".json")
# # df_tertileAll.to_json(path)
# # path = os.path.join(folderName + "/data_tertile_BP_"+unitName+".json")
# # df_tertile_BP.to_json(path)

# datHash["PDR_t1"] = re_sampling(y, int(y.shape[1]/10)).tolist()
# datHash["PDR_t2"] = re_sampling(y2, int(y2.shape[1]/10)).tolist()
# datHash["baseline"] = re_sampling(datHash["baseline"] , int(len(datHash["baseline"][0])/10)).tolist()
   
# path = os.path.join(folderName + "/data_df_"+unitName+".json")
# df.to_json(path)
# path = os.path.join(folderName + "/data_roll_"+unitName+".json")
# df_roll.to_json(path)

# with open(os.path.join(folderName + "/dataHash_"+unitName+".json"),"w") as f:
#     json.dump(dat,f)
# with open(os.path.join(folderName + "/data_MS_"+unitName+".json"),"w") as f:
#     json.dump(dat_MS_t2,f)

# with open(os.path.join(folderName+"/cfg.json"),"w") as f:
#     json.dump(cfg,f)


# %%

# import pandas as pd
# import statsmodels.formula.api as smf


# df_tmp = df[(df["target"]==4) &
#             # (df["hit_t1"]==1) &
#             (df["session"]=="Attend")]

# df_tmp_ave = df_tmp.groupby(["sub","lag","run"], sort=False, as_index=False).agg('mean', numeric_only=True)


# model = smf.mixedlm("hit_t2 ~ run", df_tmp, groups=df_tmp["sub"])
# result = model.fit()

# df_tmp["y_resid"] = result.resid




# %%

# df["sign"] = 0
# refSize = 6
# for iSub in np.unique(df["sub"]):
#     tmp = df[df["sub"]==iSub]["BaselinePD"].values
    
#     sign = []
#     for iMove in np.arange(len(tmp)):
#         if iMove < refSize+1:
#             sign.append(1)
#             continue
#         fit = np.gradient(tmp[(iMove-refSize):iMove-1])
#         weight = np.linspace(0,1,len(fit));

#         p = np.polyfit(weight,fit,3)
#         yy = np.polyval(p, np.linspace(0,1,len(fit)+1));
        
#         if tmp[iMove] > yy[-1]:
#             sign.append(1)
#         else:
#             sign.append(0)
            
#     df["sign"][df["sub"]==iSub]= np.array(sign)
    
# plt.figure()
# for iSub in np.unique(df["sub"]):
#     plt.subplot(5,4,iSub)
#     tmp = df[(df["sub"]==iSub) & (df["sign"]==0) ]
    
#     plt.plot(tmp.index,tmp["BaselinePD"],
#               marker=".", color="r")
    
#     tmp = df[(df["sub"]==iSub) & (df["sign"]==1) ]
    
#     plt.plot(tmp.index,tmp["BaselinePD"],
#               marker=".", color="b")
   
# # g = ["T1-Attend","T1-Ignored","T1-Attend","T1-Ignored"]
# df["session"][df["session"]==3] = 1
# df["session"][df["session"]==4] = 2

# df_ave = df[(df["target"]==4) & (df["res_t1"]==1)]

# df_ave = df_ave.groupby(["sub","run","session","lag","sign"], sort=False, as_index=False).agg('mean', numeric_only=True)

# df_std = df_ave.groupby(["run","session","lag","sign"]).agg(np.std)
# df_ave = df_ave.groupby(["run","session","lag","sign"], sort=False, as_index=False).agg('mean', numeric_only=True)

# # df_ave = df_ave.reset_index(drop=False)

# for iSession in np.unique(df["session"]):
#     for iSign in np.unique(df["sign"]):
#         for iLag in np.unique(df_t1["lag"]):
#             plt.plot(iLag+jitter[iSign],df_ave.loc[(iSession,slice(None),iLag,iSign),"res_t2"],col[iSign]+"o")
                              
#             plt.errorbar(iLag+jitter[iSign], df_ave.loc[(iSession,slice(None),iLag,iSign),"res_t2"],
#                       yerr = df_std.loc[(iSession,slice(None),iLag,iSign),"res_t2"]/np.sqrt(numOfSub), 
#                       markersize=10,ecolor="black")
        
# df_ave = df.groupby(["sub","run","session","sign"], sort=False, as_index=False).agg('mean', numeric_only=True)

# df_std = df_ave.groupby(["run","session","sign"]).agg(np.std)
# df_ave = df_ave.groupby(["run","session","sign"], sort=False, as_index=False).agg('mean', numeric_only=True)

# plt.figure()
# tmp = df[(df["sub"]==iSub) & (df["sign"]==0) ]

# plt.plot(tmp.index,tmp["BaselinePD"],
#           marker=".", color="r")

# tmp = df[(df["sub"]==iSub) & (df["sign"]==1) ]

# plt.plot(tmp.index,tmp["BaselinePD"],
#           marker=".", color="b")