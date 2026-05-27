#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:37:31 2025

@author: yutasuzuki
"""

# from pre_processing import split_list,split_list2

# ,rejectDat,vertical_line,rejectedByOutlier,SDT,getNearestValue
# from rejectBlink_PCA import rejectBlink_PCA

# from makeEyemetrics import makeMicroSaccade,draw_heatmap

# from joblib import Parallel, delayed
# from scipy.stats import zscore
# import itertools
# from rejectBlink_PCA import rejectBlink_PCA
# from statsmodels.stats.anova import AnovaRM
# import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp
import glob
import seaborn as sns
import msgpack
import warnings
import pandas as pd
import matplotlib
import json
from scipy.stats import zscore
import itertools
import datetime
import os
from joblib import Parallel, delayed

from pre_processing_cls import getPCPDevents,pre_processing,getNearestValue,SDT,rejectDat
from band_pass_filter import lowpass_filter

from quickpsy import quickpsy_like,fit_quickpsy_group

sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

warnings.simplefilter("ignore")

dt_now = datetime.datetime.now()
today = dt_now.strftime("%Y%m%d")

figureFolder=f"figure/{today}"
if not os.path.exists(figureFolder):
    os.makedirs(figureFolder,exist_ok=True)

#%% ------------------ data loading2 -----------------------------------------

folderName = sorted(glob.glob("./data/*"))[-1]

cfg={
"windowL":0,
"TIME_START":-2,
"TIME_END":2,
"WID_ANALYSIS":2,
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
"THRES_DIFF":1
}

f = open(folderName + "/cfg.json")
cfg.update(json.load(f))
f.close()

log = sorted(glob.glob(f"{folderName}/*_run.msgpack"))

datHash = {}
for subName in log:
    with open(subName, "rb") as f:
        tmp_datHash = msgpack.unpackb(f.read(), raw=False)

    for mmName in list(tmp_datHash.keys()):
        if mmName != "cfg":
            if not mmName in list(datHash.keys()):
                datHash[mmName] = []
            datHash[mmName] = datHash[mmName] + tmp_datHash[mmName]
    
# %%

cfg["THRES_DIFF"] = 0.3
y_corrected = np.array(datHash["PDR_trial"])
y_corrected = y_corrected.reshape(-1,y_corrected.shape[2])

# plt.plot(np.diff(y_corrected).T)
# plt.ylim([-0.25,0.25])

reject = {}
pp = pre_processing(cfg)

y,reject["PDR"] = pp.pre_processing(y_corrected)
print(reject["PDR"])

rejectArray=np.zeros(y_corrected.shape[0])
rejectArray[reject["PDR"]]=1

# %%

df = []
for iRun in np.arange(len(datHash["PDR"])):
    df.append(pd.DataFrame({
            "res_t1":np.array(datHash["res_t1"][iRun]),
            "res_t2":np.array(datHash["res_t2"][iRun]),
            "RT_t1":np.array(datHash["RT_t1"][iRun]),
            "RT_t2":np.array(datHash["RT_t2"][iRun]),
            "sub":datHash["sub"][iRun],
            "analysis":datHash["analysis"][iRun],
            "condition":datHash["condition"][iRun],
            "lag_cond":datHash["lag_cond"][iRun],
            "lag":datHash["lag"][iRun],
            "run":datHash["run"][iRun],
            }))
        
df=pd.concat(df).reset_index(drop=True)

x = np.linspace(-2, 2, y_corrected.shape[1])

df["PD"]=y_corrected[:,getNearestValue(x,-1):getNearestValue(x,0)].mean(axis=1)
df["reject"] = rejectArray

for mm in ["hit","miss","fa","cr"]:
    for t in ["_t1","_t2"]:
        df[mm+t]  = 0

mask = df["condition"].isin(["T1_only", "Both"])
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
df.loc[(df["hit_t1"]==1) & (df["hit_t2"]==1),"P(T2|T1)"] = 1

df_sum = df.groupby(["sub","analysis","condition","lag_cond"],as_index=False).agg(np.sum)
df_sum = df_sum[df_sum["condition"]=="Both"]
tmp=[]
for index, row in df_sum.iterrows():
    n = row["hit_t2"] + row["miss_t2"]
    tmp.append(SDT(row['hit_t2']/n, 0))
tmp = pd.concat(tmp)

for mmName in list(tmp.keys()):
    df_sum[mmName] = tmp[mmName].values

#%% reject sub

def remove_outliers_mad(x, k=3.5, return_z=False):
    """
    modified z-score (MAD) による外れ値除去
    x: 1D array-like
    k: threshold (default 3.5)
    """
    x = np.asarray(x, dtype=float)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    
    if med==1:
        z = zscore(x, nan_policy="omit")
        mask = np.abs(z) > 3
        
    else:
        if mad == 0 or np.isnan(mad):
            mask = np.ones_like(x, dtype=bool)
            mz = np.zeros_like(x, dtype=float)
            return (x[mask], mask, mz) if return_z else (x[mask], mask)

        mz = 0.6745 * (x - med) / mad
        mask = (np.abs(mz) > k) | np.isnan(x)
        
    return mask

# tmp_df = df[df["reject"]==0]
tmp_df = df
tmp_df = tmp_df.groupby(["sub","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)

rejectSub=[]
# for (c,h) in zip(["T1_only","T2_only"],["hit_t1","hit_t2"]):
for (c,h) in zip(["T1_only","T2_only","T2_only","T1_only"],["hit_t1","hit_t2","fa_t1","fa_t2"]):

    tmp_tmp_df = tmp_df[tmp_df["condition"]==c]

    # z = zscore(tmp_tmp_df[h], nan_policy="omit")
    # mask = np.abs(z) > 2
    
    mask = remove_outliers_mad(tmp_tmp_df[h], k=3.5, return_z=True)
    
    rejectSub.append(tmp_tmp_df[mask]["sub"].values)

    # q75, q25 = np.percentile(tmp_tmp_df[h], [75 ,25])
    # IQR = q75 - q25

    # lower = q25 - IQR*3
    # upper = q75 + IQR*3

    # sigma = np.std(tmp_tmp_df[h])    
    # lower = np.mean(tmp_tmp_df[h]) - sigma*3
    # upper = np.mean(tmp_tmp_df[h]) + sigma*3
    
    # rejectSub.append(tmp_tmp_df[(tmp_tmp_df[h]<lower)|(tmp_tmp_df[h]>upper)]["sub"].values)

rejectSub = np.concatenate(rejectSub)
rejectSub = sorted(set(rejectSub))

print(f"{rejectSub} was rejected because of low accuracy for T1 or T2")

for iSub in rejectSub:
    df = df[df["sub"]!=iSub]
    df_sum = df_sum[df_sum["sub"]!=iSub]
    
reject = [i for i,d in enumerate(datHash["sub"]) if d in rejectSub]

datHash = rejectDat(datHash,reject)

# tmp_df = df[df["reject"]==0]
# tmp_df = tmp_df.groupby(["sub","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)

# tmp_df["fa"]=0
# tmp_df.loc[tmp_df["condition"]=="T1_only","fa"] = tmp_df[tmp_df["condition"]=="T1_only"]["fa_t2"]
# tmp_df.loc[tmp_df["condition"]=="T2_only","fa"] = tmp_df[tmp_df["condition"]=="T2_only"]["fa_t1"]
# tmp_tmp_df = tmp_df[tmp_df["condition"]!="Both"]

# h="fa"
# upper = np.mean(tmp_tmp_df[h] ) + np.std(tmp_tmp_df[h])*2

# rejectSub.append(tmp_tmp_df[(tmp_tmp_df[h]>upper)]["sub"].values)

# rejectSub = np.concatenate(rejectSub)
# rejectSub = sorted(set(rejectSub))


#%% ------------------ -----------------------------------

print("pre-processing...")

y_corrected = datHash["PDR"]
pLen = 1.5
xx = np.arange(2*pLen*cfg["RESAMPLING_RATE"])*(1/cfg["RESAMPLING_RATE"])-pLen

y_lowpassed = lowpass_filter(y_corrected, 0.15, cfg["RESAMPLING_RATE"])
# y_lowpassed = lowpass_filter(y_corrected, 0.2, cfg["RESAMPLING_RATE"])
# y_lowpassed = lowpass_filter(y_corrected, 0.5, cfg["RESAMPLING_RATE"])
# y_lowpassed = y_corrected.copy()

y = [np.array(y).reshape(1,-1) for y in y_lowpassed]
y2 = [np.array(y).reshape(1,-1) for y in y_corrected]
PCPDevents = getPCPDevents(y, 0.02)
# PCPDevents = getPCPDevents(y, 0.001)

df_pupil_true=[]
for ev in ["peaks","troughs"]:
    for iSub,index in enumerate(PCPDevents[ev]):
        # pupil_mm = y_corrected[iSub].reshape(-1)
        
        pupil_mm = y2[iSub].reshape(-1)
        pupil = zscore(y2[iSub].reshape(-1))
        
        # pupil_mm = y[iSub].reshape(-1)
        # pupil = zscore(y[iSub].reshape(-1))
        
        # plt.figure();plt.plot(pupil_mm)
        
        for itmp,tmp_idx in enumerate(index):

            if (int(pLen*cfg["RESAMPLING_RATE"]) < len(pupil_mm)) & (int(tmp_idx+pLen*cfg["RESAMPLING_RATE"]) < len(pupil_mm)):
                ev_ind = np.arange(tmp_idx-int(pLen*cfg["RESAMPLING_RATE"]),int(tmp_idx+pLen*cfg["RESAMPLING_RATE"]))
                
                if len(np.argwhere(np.diff(pupil_mm[ev_ind])>1))==0:
                    
                    df_pupil_true.append(pd.DataFrame({
                    "PDR":pupil[ev_ind],
                    "PDR_mm":pupil_mm[ev_ind],
                    "sub":datHash["sub"][iSub],
                    "analysis":ev,
                    "time":xx,
                    # "count":trialCount,
                    # "timeFromEv":xx[np.argmin(pupil[ev_ind])] if datHash["analysis"][i]=="LBP" else xx[np.argmax(pupil[ev_ind])] ,
                    }))
    
df_pupil_true = pd.concat(df_pupil_true).reset_index(drop=True)
df_pupil_true = df_pupil_true.groupby(["sub","analysis","time"],sort=False,as_index=False).agg("mean",numeric_only=True)

g = sns.FacetGrid(df_pupil_true, 
                  # col="sub",
                  # col_wrap=5,
                  hue="analysis",
                  )
g.map(sns.lineplot, "time", "PDR",errorbar="se").add_legend()
# g.map(sns.lineplot, "time", "PDR_mm",errorbar="se").add_legend()

# %%

from scipy.stats import linregress
from scipy.signal import butter, filtfilt, hilbert

pLen = 1.5
xx = np.arange(2*pLen*cfg["RESAMPLING_RATE"])*(1/cfg["RESAMPLING_RATE"])-pLen

tmp_pupil={"LBP":[],"SBP":[]}
df_pupil=[]
df_pupilBP=[]
df_F1=[]

detectwin=5
trialCount=0
for i,condition in enumerate(datHash["t1"]):

    # pupil = zscore(y[i].reshape(-1))
    # tmp_y = np.array(y_lowpassed[i])

    pupil = zscore(y2[i].reshape(-1))
    tmp_y = np.array(y2[i])

    # pupil_mm = y[i].reshape(-1)
    pupil_mm = y2[i].reshape(-1)

    analytic = hilbert(pupil_mm)
    phase = np.angle(analytic)

    x = np.arange(len(tmp_y))*(1/cfg["RESAMPLING_RATE"])
    # aaa
    # plt.plot(x,tmp_y)
    # pcpd = np.array(PCPDevents["peaks"][i])
    # # pcpd = np.array(PCPDevents["troughs"][i])
    # plt.plot(x[pcpd],tmp_y[pcpd],'o')
    
    # tmp_condition = np.array(condition[:-1])
    # tmp_condition = tmp_condition*(cfg["RESAMPLING_RATE"]/cfg["SAMPLING_RATE"])
    # tmp_condition = tmp_condition.astype(int)

    # plt.plot(x[tmp_condition],tmp_y[tmp_condition],'o')
    # plt.xlim([200,300])

    for j,idx in enumerate(condition[:-1]):
        
        tmp_idx = int(idx*(cfg["RESAMPLING_RATE"]/cfg["SAMPLING_RATE"]))
        
        if datHash["analysis"][i]=="LBP":
            tmp_ev = np.array(PCPDevents["peaks"][i])
        else:
            tmp_ev = np.array(PCPDevents["troughs"][i])
        
        df_F1.append(pd.DataFrame({
            "sub":datHash["sub"][i],
            "analysis":datHash["analysis"][i],
            "count":j,
            "F1":0,
            "delta":np.nan
            },index=[0]))
        
        cond = tmp_ev[(tmp_ev > tmp_idx-(0*cfg["RESAMPLING_RATE"])) & 
                      (tmp_ev < tmp_idx+(detectwin*cfg["RESAMPLING_RATE"]))]
       
        if len(cond) > 0:
            df_F1[-1]["F1"] = 1
            df_F1[-1]["delta"] = (cond[0]-tmp_idx)/cfg["RESAMPLING_RATE"]

        ev_ind = np.arange(tmp_idx-int(pLen*cfg["RESAMPLING_RATE"]),int(tmp_idx+pLen*cfg["RESAMPLING_RATE"]))
        if len(np.argwhere(abs(np.diff(pupil_mm[ev_ind]))>0.1))==0:
            
            slope, intercept, r, pval, stderr = linregress(xx, pupil_mm[ev_ind])
                        
            df_pupil.append(pd.DataFrame({
                "PDR":pupil[ev_ind],
                "PDR_mm":pupil_mm[ev_ind],
                "sub":datHash["sub"][i],
                "analysis":datHash["analysis"][i],
                "time":xx,
                "count":trialCount,
                "timeFromEv":xx[np.argmin(pupil[ev_ind])] if datHash["analysis"][i]=="LBP" else xx[np.argmax(pupil[ev_ind])] ,
                }))
            trialCount+=1
            
            df_pupilBP.append(pd.DataFrame({
                "PDR_bp":pupil_mm[np.arange(tmp_idx-1*cfg["RESAMPLING_RATE"],tmp_idx)].mean(),
                "sub":datHash["sub"][i],
                "analysis":datHash["analysis"][i],
                "condition":datHash["condition"][i][j],
                "slope":slope,
                "phase":np.angle(np.mean(np.exp(1j * phase[tmp_idx-int(0.1*cfg["RESAMPLING_RATE"]):tmp_idx+int(0.1*cfg["RESAMPLING_RATE"])])))
                },index=[0]))
        
df_pupil = pd.concat(df_pupil).reset_index(drop=True)
df_pupilBP = pd.concat(df_pupilBP).reset_index(drop=True)
df_F1 = pd.concat(df_F1).reset_index(drop=True)

# df_F1_ave = df_F1.groupby(["sub","analysis"],sort=False,as_index=False).agg(np.nanmean)

# g = sns.FacetGrid(df_F1_ave, 
#                   hue="analysis",
#                   )
# g.map(sns.pointplot, "analysis", "F1",errorbar="se").add_legend()


# g = sns.FacetGrid(df_F1_ave, 
#                   hue="analysis",
#                   )
# g.map(sns.pointplot, "analysis", "delta",errorbar="se").add_legend()


tmp_df = df_pupil.groupby(["sub","analysis"],sort=False,as_index=False).agg("mean",numeric_only=True)

plt.figure()
sns.pointplot(
    data=tmp_df,
    x="analysis", 
    y="PDR_mm",
    # y="PDR",
    errorbar="se"
)

tmp_df = df_pupil.groupby(["sub","analysis","time"],sort=False,as_index=False).agg("mean",numeric_only=True)

g = sns.FacetGrid(tmp_df, 
                  # col="sub",
                  # col_wrap=5,
                  hue="analysis",
                  # hue="lag_cond",
                  )
# g.map(sns.lineplot, "time", "PDR",errorbar="se").add_legend()
g.map(sns.lineplot, "time", "PDR_mm",errorbar="se").add_legend()

df_pupilBP_ave = df_pupilBP.groupby(["sub","analysis"],sort=False,as_index=False).agg("mean",numeric_only=True)

def circmean(x):
    return np.angle(np.mean(np.exp(1j * x)))

df_pupilBP_ave = df_pupilBP.groupby(
    ["sub","analysis"],
    sort=False,
    as_index=False
).agg({
    "phase": circmean,
    "slope": "mean"
})
plt.figure()
sns.pointplot(
    data=df_pupilBP_ave,
    x="analysis", 
    # y="slope",
    y="phase",
    errorbar="se"
)


# g = sns.FacetGrid(df_pupil, 
#                   col="analysis",
#                   # col_wrap=5,
#                   hue="count",
#                   # hue="lag_cond",
#                   )
# # g.map(sns.lineplot, "time", "PDR",errorbar="se").add_legend()
# # g.map(sns.lineplot, "time", "PDR_mm",errorbar="se").add_legend()
# g.map(sns.lineplot, "time", "PDR_mm",errorbar=None).add_legend()


# %% pupil change around target onset

def pick_peak(g):
    t = g["time"].to_numpy()
    v = g["PDR"].to_numpy()

    if g["analysis"].iat[0] == "LBP":
        idx = np.argmax(v[t<0])
    else:
        idx = np.argmin(v[t<0])

    peak_t = float(t[idx])
    peak_v = float(v[idx])

    return pd.Series({
        "peak_time": peak_t,
        "peak_value": peak_v,
    })


peak_df = (
    tmp_df
    .groupby(["sub", "analysis"], sort=False)
    .apply(pick_peak)
    .reset_index()
)

sns.pointplot(
    data=peak_df,
    x="analysis", 
    y="peak_time",
    errorbar="se"
)


figcount = 0

plt.savefig(f"{figureFolder}/{figcount}_PDR_around_events.png")
figcount+=1

tmp_df = df_pupil.groupby(["sub","analysis"],sort=False,as_index=False).agg("mean",numeric_only=True)
g = sns.FacetGrid(tmp_df, 
                  # col="sub",
                  # hue="analysis",
                  # hue="lag_cond",
                  )
g.map(sns.pointplot, "analysis", "PDR_mm",errorbar="se").add_legend()
# g.map(sns.pointplot, "analysis", "PDR",errorbar="se").add_legend()
plt.savefig(f"{figureFolder}/{figcount}_average_PDR.png")
figcount+=1

tmp_df = df_pupilBP.copy()
tmp_df = tmp_df[tmp_df["condition"]=="Both"]
tmp_df = tmp_df.groupby(["sub","analysis"],sort=False,as_index=False).agg("mean",numeric_only=True)

tmp_df["pupil_diff"]=0
for iSub in tmp_df["sub"].unique():
    
    tmp_tmp_df = tmp_df[(tmp_df["sub"]==iSub)]
    
    if tmp_tmp_df[tmp_tmp_df["analysis"]=="LBP"]["PDR_bp"].values[0] > tmp_tmp_df[tmp_tmp_df["analysis"]=="SBP"]["PDR_bp"].values[0]:
        tmp_df.loc[(tmp_df["sub"]==iSub),"pupil_diff"]=1

plt.figure(figsize=(4, 6))
sns.pointplot(
    data=tmp_df,
    x="analysis", 
    y="PDR_bp",
    # hue="pupil_diff",
    # style="sub",
    # markers=True,
    errorbar="se"
    # legend=False
)
plt.savefig(f"{figureFolder}/{figcount}_PDR_BP.png")
figcount+=1


# %%

mmName = [
    "amp_peaks_st",
    "amp_troughs_st",
    "wave_onset_peaks_st",
    "wave_onset_troughs_st",
    ]

pLen = 3

wave_summary = []
wave_summary_actual = []
pupil_pktr=[]

events={"SBP":[],"LBP":[]}

# plt.figure()
for i,condition in enumerate(datHash["t2"]):
    
    pcpd = PCPDevents["summary"][i].copy()
    pupil = zscore(y[i].reshape(-1))
    
    time = np.arange(len(pcpd)) * 1/cfg["RESAMPLING_RATE"]

    idx={"trough":{},"peak":{},
         "amp_peak_st":[],"amp_trough_st":[],
         "wave_peak_st":[],"wave_trough_st":[],
         }
    for j in range(len(condition)):
        
        onset_idx = round(condition[j]/int(cfg["SAMPLING_RATE"]/cfg["RESAMPLING_RATE"]))
        
        idx["peak"]["back"] = onset_idx
        
        ## label: peak = 1
        while (pcpd[idx["peak"]["back"]]<=0) & (idx["peak"]["back"]>0):
            idx["peak"]["back"] -= 1

        idx["trough"]["back"] = onset_idx
        
        ## label: troughs = -1
        while (pcpd[idx["trough"]["back"]]>=0) & (idx["trough"]["back"]>0):
            idx["trough"]["back"] -= 1
        
        for mmName2 in ["trough","peak"]:
            if (len(pupil) > idx[mmName2]["back"]+pLen*cfg["RESAMPLING_RATE"])&(len(pupil) > idx[mmName2]["back"]+pLen*cfg["RESAMPLING_RATE"]):
                # if len(np.argwhere(abs(np.diff(pupil_troughs)) > b*3))==0 & len(np.argwhere(abs(np.diff(pupil_peaks)) > b*3))==0:
                    
                pupil_pktr.append(pd.DataFrame({
                    "pupil":pupil[np.arange(idx[mmName2]["back"]-pLen*cfg["RESAMPLING_RATE"],idx[mmName2]["back"]+pLen*cfg["RESAMPLING_RATE"])],
                    "sub":datHash["sub"][i],
                    "session":datHash["analysis"][i],
                    "Run":datHash["run"][i],
                    "analysis":mmName2,
                    "Time":np.arange(pLen*2*cfg["RESAMPLING_RATE"])*(1/cfg["RESAMPLING_RATE"])-pLen
                    }))
                
            idx[f"amp_{mmName2}_st"].append(np.round(pupil[idx[mmName2]["back"]] - pupil[onset_idx], 4))
            idx[f"wave_{mmName2}_st"].append(np.round(time[onset_idx] - time[idx[mmName2]["back"]], 4))

    onset_idx = (np.array(condition) / (cfg["SAMPLING_RATE"] / cfg["RESAMPLING_RATE"])).astype(int)
    
    events["SBP"].append(np.diff(time[np.array(pcpd)==-1]))
    events["LBP"].append(np.diff(time[np.array(pcpd)==1]))
    
    wave_summary_actual.append(pd.DataFrame({
        "dat":np.diff(time[np.array(pcpd)==-1]),
        "label":"troughs",
        "sub":datHash["sub"][i],
        "run":datHash["run"][i],
        "session":datHash["analysis"][i],
        }))
    
    wave_summary_actual.append(pd.DataFrame({
        "dat":np.diff(time[np.array(pcpd)==1]),
        "label":"peaks",
        "sub":datHash["sub"][i],
        "run":datHash["run"][i],
        "session":datHash["analysis"][i],
        }))
    
    
    wave_summary.append(pd.DataFrame({
        "onset":np.diff(np.r_[0,time[onset_idx]]),
        "amp_peaks_st":idx["amp_peak_st"],
        "amp_troughs_st":idx["amp_trough_st"],
        "wave_onset_peaks_st":idx["wave_peak_st"],
        "wave_onset_troughs_st":idx["wave_trough_st"],
        "sub":datHash["sub"][i],
        "run":datHash["run"][i],
        "session":datHash["analysis"][i],
        }))
    
wave_summary_actual = pd.concat(wave_summary_actual).reset_index(drop=True)
wave_summary = pd.concat(wave_summary).reset_index(drop=True)

pupil_pktr = pd.concat(pupil_pktr).reset_index(drop=True)
pupil_pktr_ave = pupil_pktr.groupby(["sub","analysis","Time"],sort=False,as_index=False).agg("mean",numeric_only=True)

# g = sns.FacetGrid(pupil_pktr_ave, 
#                   hue="analysis",
#                   )
# g.map(sns.lineplot, "Time", "pupil",errorbar="se").add_legend()

# plt.hist(list(itertools.chain.from_iterable(events["trough"])),bins=100)
# plt.hist(list(itertools.chain.from_iterable(events["peak"])),stat="density",bins=100)
# plt.hist(wave_summary[wave_summary["session"]=="LBP"]["onset"],bins=100)

# %%

bin_width = 1
bins = np.arange(2, 20, bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure()

df_hist=[]
for analysisType in ["SBP","LBP"]:
    for iSub in sorted(wave_summary["sub"].unique()):
        
        dat_actual = wave_summary_actual[
            (wave_summary_actual["sub"] == iSub) &
            (wave_summary_actual["session"] == analysisType)
        ]["dat"]

        hist_actual, _ = np.histogram(dat_actual, bins=bins)
        
        df_hist.append(pd.DataFrame({
                "sub": iSub,
                "bin_center": bin_centers,
                "density": hist_actual/ np.sum(hist_actual),
                "type": "actual",
                "session": analysisType
            }))
        
        dat_rt = wave_summary[
            (wave_summary["sub"] == iSub) &
            (wave_summary["session"] == analysisType)
        ]["onset"]

        hist_rt, _ = np.histogram(dat_rt, bins=bins)
        
        df_hist.append(pd.DataFrame({
                "sub": iSub,
                "bin_center": bin_centers,
                "density": hist_rt/np.sum(hist_rt),
                "type": "realtime",
                "session": analysisType
            }))
        
df_hist = pd.concat(df_hist)

g = sns.FacetGrid(
    df_hist,
    col="session",
    # row="sub",
    hue="type"
)
g.map(sns.lineplot, "bin_center", "density").add_legend()
g.set(
      xlabel="time diff fron previous event",
      ylabel="Value",
      )

plt.savefig(f"{figureFolder}/{figcount}_hist.png")
figcount+=1

# %%

for mmName2 in ["wave_onset_troughs_st","wave_onset_peaks_st"]:
    g = sns.FacetGrid(wave_summary[(wave_summary[mmName2]<10)],
                      # col_wrap=5,
                      # col="sub",
                      hue="session",
                      )
    g.map(sns.histplot, 
          mmName2,
          kde=True,
          stat="density",
          bins=100
          ).add_legend()
    g.set_titles(mmName2)
    
    plt.savefig(f"{figureFolder}/{figcount}_hist_{mmName2}.png")
    figcount+=1


# for ax,colName in zip(g.axes.flat,list(g.axes_dict.keys())):
#     ax.axvline(x=10,color="k")


# %% regressout run effect

df_tmp = df[((df["condition"]=="T2_only")|(df["condition"]=="Both"))]
df_tmp["y"] = df_tmp["hit_t2"]

df_tmp[["y_resid","RT_t2_resid"]] = np.nan
df[["y_resid",
    "y_resid_PD",
    "RT_t2_resid",
    "y_resid_smooth_norm",
    "P(T2|T1)_smooth_norm",
    "RT_t2_resid_norm","time_bin"]] = np.nan

for iSession in df_tmp["analysis"].unique():
    for iTarget in df_tmp["condition"].unique():
        
        df_tmp_tmp = df_tmp[(df_tmp["analysis"]==iSession)&
                            (df_tmp["condition"]==iTarget)]

        model = smf.mixedlm("y ~ run", df_tmp_tmp, groups=df_tmp_tmp["sub"])
        result = model.fit()
        
        df_tmp.loc[(df_tmp["analysis"]==iSession)&
                   (df_tmp["condition"]==iTarget),"y_resid"] = result.resid + result.params["Intercept"]

        df.loc[(df["analysis"]==iSession)&
               (df["condition"]==iTarget),"y_resid"] = result.resid + result.params["Intercept"]
        

df_tmp = df_tmp[(df_tmp["condition"]=="Both")]

model = smf.mixedlm("y ~ PD", df_tmp, groups=df_tmp["sub"])
result = model.fit()

df_tmp["y_pred"] = result.predict(df_tmp)
df.loc[df["condition"]=="Both","y_resid_PD"]  = df_tmp["y"] - df_tmp["y_pred"] + df_tmp["y"].mean()

# df.loc[df["condition"]=="Both","y_resid_PD"] = result.resid

# for iSession in df_tmp["analysis"].unique():
 
#         df_tmp_tmp = df_tmp[(df_tmp["analysis"]==iSession)&
#                             (df_tmp["condition"]==iTarget)&
#                             (df_tmp["RT_t2"]!=-1)]

#         model = smf.mixedlm("RT_t2 ~ run", df_tmp_tmp, groups=df_tmp_tmp["sub"])
#         result = model.fit()

#         df.loc[(df["analysis"]==iSession)&
#                (df["condition"]==iTarget)&
#                (df["RT_t2"]!=-1),"RT_t2_resid"] = result.resid + result.params["Intercept"]
        
#         df_tmp.loc[(df_tmp["analysis"]==iSession)&
#                    (df_tmp["condition"]==iTarget)&
#                    (df_tmp["RT_t2"]!=-1),"RT_t2_resid"] = result.resid + result.params["Intercept"]


df[mmName]=0
for iSub in df["sub"].unique():
    for iRun in df["run"].unique():
        for iSession in df["analysis"].unique():
            for m in mmName:
                df.loc[(df["sub"]==iSub)&
                       ((df["condition"]=="T2_only")|(df["condition"]=="Both"))&
                       (df["analysis"]==iSession)&
                       (df["run"]==iRun),m] = wave_summary[(wave_summary["sub"]==iSub)&
                                                           (wave_summary["session"]==iSession)&
                                                           (wave_summary["run"]==iRun)][m].values

# %% order effect regressout

sTime = 0
eTime = 11
jitter=1

# sTime = -0.1
# eTime = 0.15
# jitter=0.05

target = "_pupil"
# event = "_peaks"
event = "_troughs"
df_tmp=df.copy()
df_tmp["wave_onset"] = df_tmp[f"wave_onset{event}_st"]
df["wave_onset"] = df[f"wave_onset{event}_st"]

# df_tmp["wave_onset"] = df_tmp[f"amp{event}_st"]
# df["wave_onset"] = df[f"amp{event}_st"]


df_tmp = df_tmp[(df_tmp["wave_onset"]<=eTime)]

df_tmp_tmp = df_tmp.copy()
df_tmp_tmp["P(T2|T1)_smooth_norm"] = np.nan

for iSub in df_tmp_tmp["sub"].unique():
    for iSession in df_tmp_tmp["analysis"].unique():
        for iLag in df_tmp_tmp["lag_cond"].unique():
        # for iCond in df_tmp_tmp["condition"].unique():

            ind = (df_tmp_tmp["sub"]==iSub)&(df_tmp_tmp["analysis"]==iSession)&(df_tmp_tmp["lag_cond"]==iLag)
            
            t = df_tmp_tmp[ind]["y_resid"].values
            # if np.isnan(t[0] / t.mean()) ==True:
            #     df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = 0
            # else:
                # df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = t / t.mean() * 100 -100
    
            df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = t - np.nanmean(t[t!=-100])
            
            t = df_tmp_tmp[ind]["P(T2|T1)"].values
            df_tmp_tmp.loc[ind,"P(T2|T1)_smooth_norm"] = t - np.nanmean(t[t!=-100])
            
            t = df_tmp_tmp[ind]["RT_t2_resid"].values
            df_tmp_tmp.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t[t!=-100])
            
            for iTarget in df_tmp_tmp["condition"].unique():
                
                ind = (df["sub"]==iSub)&(df["analysis"]==iSession)&(df["wave_onset"]<=eTime)&(df["condition"]==iTarget)&(df["lag_cond"]==iLag)
                       
                t = df[ind]["y_resid"].values
                df.loc[ind,"y_resid_smooth_norm"] = t - np.nanmean(t[t!=-100])
                # if np.isnan(t[0] / t.mean()) ==True:
                #     df.loc[ind,"y_resid_smooth_norm"] = 0          
                # else:
                #     df.loc[ind,"y_resid_smooth_norm"] = t / t.mean() * 100 -100
                #     print(f"{iSession},{iLag},{iTarget}")
                #     print(t / t.mean() * 100 - 100)
                            
                t = df[ind]["P(T2|T1)"].values
                df.loc[ind,"P(T2|T1)_smooth_norm"] = t - np.nanmean(t[t!=-100])
    
                t = df[ind]["RT_t2_resid"].values            
                df.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t[t!=-100])
                # df.loc[ind,"RT_t2_resid_norm"] = t / np.nanmean(t)* 100
        

df_timebin=[]
df_tmp_tmp["time_bin"] = np.nan

for iTime in np.arange(sTime,eTime-jitter,jitter):

    t = df_tmp_tmp[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter)].copy()
    df_tmp_tmp.loc[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter),"time_bin"] = int(iTime)
    
    df.loc[(df["wave_onset"]>iTime)&(df["wave_onset"]<iTime+jitter),"time_bin"] = int(iTime)
    
    t = t.groupby(["sub","analysis","lag_cond"],sort=False,as_index=False).agg("mean",numeric_only=True)
    # del t["analysis"]
    # t = t.groupby(["sub","lag_cond"],sort=False,as_index=False).agg(np.nanmean)
    t["time_bin"] = int(iTime)
    df_timebin.append(t)
    
df_timebin = pd.concat(df_timebin).reset_index(drop=True)

# %% wave time bin

df_timebin = df.copy()
df_timebin = df_timebin[df_timebin["time_bin"]>=0]

df_timebin = df_timebin.groupby(["sub","analysis","time_bin"],sort=False,as_index=False).agg("mean",numeric_only=True)

for iSession in df_timebin["analysis"].unique():

    df_tmp = df_timebin[(df_timebin["analysis"]==iSession)]

    df_pval=[]
    for iTime in sorted(df_timebin["time_bin"].unique()):

        t = df_tmp[
            # (df_tmp["lag_cond"]==iLag)&
            (df_tmp["time_bin"]==iTime)
            ]

        t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=0)

        # t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=1)
        # t_stat, p_two_sided = ttest_1samp(t["P(T2|T1)_smooth_norm"], popmean=0)

        df_pval.append(pd.DataFrame({
            # "lag_cond":iLag,
            "time":iTime,
            "tval":t_stat,
            "pval":round(p_two_sided,4),
            },index=[0]))
            
    print(f"{iSession}:{df_pval}")
    
    # df_pval = pd.concat(df_pval)
    


# %% false alarm

df = df[df["reject"]==0]
df_ave = df.groupby(["sub","analysis","lag_cond","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)


df_plot=df_ave[df_ave["condition"]=="Both"]

plt.figure()
g = sns.FacetGrid(df_plot,
                  # col="sub",
                  # col_wrap=5,
                  hue="analysis",
                  # hue="lag_cond",
                  )
g.map(sns.pointplot, 
      "lag_cond", 
      "P(T2|T1)",
      # "y_resid_PD",
      # "y_resid",
      errorbar="se").add_legend()

# g.map(sns.pointplot, "lag_cond", "P(T2|T1)_smooth_norm",errorbar="se").add_legend()
# g.map(sns.pointplot, "lag_cond", "y_resid",errorbar="se").add_legend()

for ax,colName in zip(g.axes.flat,list(g.axes_dict.keys())):
    ax.axhline(y=0.5,color="k")

plt.savefig(f"{figureFolder}/{figcount}_P(T2T1).png")
figcount+=1


df_ave["res"]=0
df_ave.loc[df_ave["condition"]=="T1_only","res"]=df_ave[df_ave["condition"]=="T1_only"]["res_t1"]
df_ave.loc[df_ave["condition"]=="T2_only","res"]=df_ave[df_ave["condition"]=="T2_only"]["res_t2"]

plt.figure()
g = sns.FacetGrid(df_ave[df_ave["condition"]!="Both"], 
                  row="condition",
                  # col_wrap=5,
                  # hue="analysis",
                  # hue="lag_cond",
                  )
g.map(sns.pointplot, "sub", "res",errorbar="se").add_legend()

for ax,colName in zip(g.axes.flat,list(g.axes_dict.keys())):
    ax.axhline(y=0.5,color="k")

g.fig.set_figheight(4)
g.fig.set_figwidth(15)

plt.savefig(f"{figureFolder}/{figcount}_T1T2only.png")
figcount+=1


df_ave["fa"]=0
df_ave.loc[df_ave["condition"]=="T1_only","fa"] = df_ave[df_ave["condition"]=="T1_only"]["fa_t2"]
df_ave.loc[df_ave["condition"]=="T2_only","fa"] = df_ave[df_ave["condition"]=="T2_only"]["fa_t1"]

df_ave_ave = df_ave.groupby(["sub","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)

plt.figure()
g = sns.FacetGrid(df_ave_ave[df_ave_ave["condition"]!="Both"], 
                  # col="sub",
                  # col_wrap=5,
                  # hue="analysis",
                  # hue="lag_cond",
                  )
g.map(sns.pointplot, "sub", "fa",errorbar="se").add_legend()
g.fig.set_figheight(4)
g.fig.set_figwidth(10)

plt.savefig(f"{figureFolder}/{figcount}_FA.png")
figcount+=1

# # %%
# plt.figure()
# g = sns.FacetGrid(df, 
#                   col="sub",
#                   row="analysis",
#                   # hue="lag_cond",
#                   )
# # g.map(sns.pointplot, "time_bin", "y_resid_smooth_norm",
# g.map(sns.scatterplot, "wave_onset", "P(T2|T1)_smooth_norm",
#       # errorbar=None
#       # errorbar="ci"
#       # errorbar="se"
#       )
# g.set(xlim=[-1,1])
# g.set(ylim=[-1,1])


# %% wave accuracy

df_timebin = df.copy()
df_timebin = df_timebin[df_timebin["time_bin"]>=0]

df_timebin = df_timebin.groupby(["sub","analysis","time_bin"],sort=False,as_index=False).agg("mean",numeric_only=True)

plt.figure()
g = sns.FacetGrid(df_timebin, 
                  # col="lag_cond",
                  row="analysis",
                  # hue="lag_cond",
                  )
# g.map(sns.pointplot, "time_bin", "y_resid_smooth_norm",
g.map(sns.pointplot, "time_bin", "P(T2|T1)_smooth_norm",
      # errorbar=None
      errorbar="ci"
      # errorbar="se"
      ).add_legend()
g.fig.set_figheight(8)
g.fig.set_figwidth(10)
# g.fig.set_figwidth(8)
# plt.axhline(y=0,color="k")
for ax,colName in zip(g.axes.flat,list(g.axes_dict.keys())):
    ax.axhline(y=0,color="k")

plt.savefig(f"{figureFolder}/{figcount}_accuracy.png")
figcount+=1

    
# %% accuracy regressout

df_timebin = df.copy()
df_timebin = df_timebin[df_timebin["time_bin"]>=0]

df_timebin = df_timebin.groupby(["sub","analysis","time_bin","lag_cond"],sort=False,as_index=False).agg("mean",numeric_only=True)


for iSession in df_timebin["analysis"].unique():
    
    df_tmp = df_timebin[(df_timebin["analysis"]==iSession)]

    df_pval=[]
    for iLag in df_tmp["lag_cond"].unique():
        for iTime in df_tmp["time_bin"].unique():
    
            t = df_tmp[(df_tmp["lag_cond"]==iLag)&
                       (df_tmp["time_bin"]==iTime)]
    
            # t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=0)
            # t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=1)
            t_stat, p_two_sided = ttest_1samp(t["P(T2|T1)_smooth_norm"], popmean=0)
            # t_stat, p_two_sided = ttest_1samp(t["P(T2|T1)"], popmean=0)
    
            df_pval.append(pd.DataFrame({
                "lag_cond":iLag,
                "time":iTime,
                "tval":t_stat,
                "pval":round(p_two_sided,4),
                },index=[0]))
            
    df_pval = pd.concat(df_pval)
    
    df_pval['annot'] = df_pval['pval'].apply(lambda p: '*' if p < 0.05 else '')
    # annot_table = df_pval.pivot(index='lag', columns='time', values='pval')
    annot_table = df_pval.pivot(index='lag_cond', columns='time', values='annot')
    
    pivot_table = df_tmp.pivot_table(
        index='lag_cond',
        columns='time_bin',
        # values='y_resid_smooth_norm'
        values='P(T2|T1)_smooth_norm'
        # values='P(T2|T1)'
    )
    
    pivot_table.columns = (pivot_table.columns).astype(int)
    
    plt.figure()
    ax=sns.heatmap(pivot_table, 
                cmap='RdBu_r', 
                cbar=True,
                annot=annot_table,
                fmt='',
                center=0,
                # annot_kws={"size": 20, "color": "black","weight": "bold"}
                annot_kws={"size": 5, "color": "black","weight": "bold"}
                )
    ax.set_title(f"Method={iSession}")
    ax.set_xlabel("Time[s]")
    ax.set_ylabel("lag_cond")
    ax.invert_yaxis()
    
    plt.savefig(f"{figureFolder}/{figcount}_accuracy_{iSession}.png")
    figcount+=1

# %% pupil size

def groupwise_qcut(x):
    try:
        return pd.qcut(x, q=5, labels=False, duplicates='drop')
    except ValueError:
        return pd.Series([np.nan] * len(x), index=x.index)

df['PD_bin'] = (
    df.groupby(['sub', 'analysis', 'lag_cond','condition'])['PD']
    .transform(groupwise_qcut)
)

# %% psychometric func.

group_cols = ["sub", "analysis"]
groups = list(df.groupby(group_cols))

group_cols = ["analysis"]
groups = list(df.groupby(group_cols))

def run_one_group(key, g):
    # sub, analysis = key
    analysis = key
    
    out = fit_quickpsy_group(
        g,
        x_col="lag_cond",
        y01_col="P(T2|T1)",
        fun_name="cum_normal",
        # grouping_values={"sub": sub, "analysis": analysis},
        grouping_values={"analysis": analysis},
        prob=0.5,
        bootstrap="parametric",
        # bootstrap="non-parametric",
        B=10000,
        seed=42
    )
    return out

outs = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
    delayed(run_one_group)(key, g.copy()) for key, g in groups
)

outs = [o for o in outs if o is not None]

# fit_sub = {
#     "par": pd.concat([o["par"] for o in outs], ignore_index=True),
#     "parci": pd.concat([o["parci"] for o in outs], ignore_index=True),
#     "thresholds": pd.concat([o["thresholds"] for o in outs], ignore_index=True),
#     "thresholdsci": pd.concat([o["thresholdsci"] for o in outs], ignore_index=True),
#     "curves": pd.concat([o["curves"] for o in outs], ignore_index=True),
#     "data": pd.concat([o["data"] for o in outs], ignore_index=True),
#     "fun": "cum_normal",
#     "prob": 0.75,
#     "bootstrap": "parametric",
#     "B": 1000
# }

boot_thr = []
for o in outs:
    # sub = o["thresholds"].iloc[0]["sub"]
    analysis = o["thresholds"].iloc[0]["analysis"]
    prob = o["thresholds"].iloc[0]["prob"]

    arr = o.get("boot_thresholds", None)
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]

    boot_thr.append(pd.DataFrame({
        # "sub": sub,
        "threshold_boot": arr,
        "thresholds": o["thresholds"]["threshold"].values[0],
        "analysis": analysis[0],
        "prob": prob
    }))

boot_thr_df = pd.concat(boot_thr, ignore_index=True) if boot_thr else pd.DataFrame()

g = sns.displot(
    data=boot_thr_df,
    x="threshold_boot",
    hue="analysis",
    bins=40,
    common_bins=True
)
g.set_axis_labels("threshold", "count")
plt.show()

# %%
normFlg=""
df.to_json(f"{folderName}/df{normFlg}.json")
df_ave.to_json(f"{folderName}/df_ave{normFlg}.json")
# pupil_pktr.to_json(f"{folderName}/df_pupilTimeCourse{normFlg}.json")
df_sum.to_json(f"{folderName}/df_sum{normFlg}.json")

boot_thr_df.to_json(f"{folderName}/df_pse_thre.json")

# df_pupil = df_pupil[df_pupil["target"]>2]
df_pupil_ave = df_pupil.groupby(["sub","analysis","time"],sort=False,as_index=False).agg("mean",numeric_only=True)

df_pupil_ave.to_json(f"{folderName}/df_pupil{normFlg}.json")
df_timebin.to_json(f"{folderName}/df_timebin{normFlg}.json")
df_pupilBP_ave.to_json(f"{folderName}/df_pupilBP_ave{normFlg}.json")
