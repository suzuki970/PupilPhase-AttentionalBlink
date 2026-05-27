#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 11:13:56 2025

@author: yutasuzuki
"""

import numpy as np
import matplotlib.pyplot as plt


from pre_processing import split_list,split_list2
from pre_processing_cls import pre_processing,rejectDat,vertical_line,rejectedByOutlier,SDT,getNearestValue
from pre_processing_cls import getPCPDevents,getfft_multi,morlet_cwt_multi
from band_pass_filter import lowpass_filter
# from rejectBlink_PCA import rejectBlink_PCA
import json
import os
import warnings
import pandas as pd
import matplotlib
# from makeEyemetrics import makeMicroSaccade,draw_heatmap
import datetime
import glob
import seaborn as sns
import msgpack
from joblib import Parallel, delayed
from scipy.stats import zscore
from tqdm import tqdm
import itertools
from scipy.stats import ttest_1samp
from rejectBlink_PCA import rejectBlink_PCA
from statsmodels.stats.anova import AnovaRM
from scipy.stats import norm
import statsmodels.formula.api as smf
import matplotlib.gridspec as gridspec


sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

warnings.simplefilter("ignore")

# %%
bpass=0.15
# def run(bpass):

#% ------------------ initial settings ------------------

cfg={
"windowL":0,
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

folderName = sorted(glob.glob(os.path.join("./data/*")))[-1]

savefolder = f"./data/{date}"

if not os.path.exists(savefolder):
    os.mkdir(savefolder)


f = open(folderName + "/cfg.json")
cfg.update(json.load(f))
f.close()

print("data loding...")

# cfg["normFlag"]=True
cfg["normFlag"]=False

if cfg["normFlag"]:
    normFlg="_norm"
else:
    normFlg=""
    
    
log = sorted(glob.glob(f"./data/**/*_trial{normFlg}.msgpack"))[-1].split('/')[2]
log = sorted(glob.glob(f"./data/{log}/*_trial{normFlg}.msgpack"))
    
datHash = {}
for subName in log:
    with open(subName, "rb") as f:
        tmp_datHash = msgpack.unpackb(f.read(), raw=False)

    for mmName in list(tmp_datHash.keys()):
        if mmName != "cfg":
            if not mmName in list(datHash.keys()):
                datHash[mmName] = []
            datHash[mmName] = datHash[mmName] + tmp_datHash[mmName]
    
time_x = np.linspace(cfg["TIME_START"], cfg["TIME_END"],int((cfg["TIME_END"]-cfg["TIME_START"])*cfg["RESAMPLING_RATE"]))

df = pd.DataFrame()
# for mmName in ["sub","target","lag","run","time"]:
for mmName in ["sub","session","lag","run","time"]:
    df[mmName] = datHash[mmName]

g = ["no_sound","T1_only","T2_only","Both"]
df["condition"] = [g[int(t-1)] for t in datHash["target"]]

y_bp = np.array(datHash["baseline"])
x = np.linspace(-cfg["WID_BP_ANALYSIS"], 0, y_bp.shape[1])

y_bp = y_bp[:,getNearestValue(x,-1):getNearestValue(x,0)]

df["PD"] = y_bp.mean(axis=1)
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

#### Target 1:no sound, 2:T1 only, 3:T2 only, 4:Both
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
df["P(T2|T1)"][(df["hit_t1"]==1)&(df["hit_t2"]==1)&(df["session"]=="T1-active")]=1
df["P(T2|T1)"][(df["hit_t2"]==1)&(df["session"]=="T1-passive")]=1

#%% reject sub

tmp_df = df[(df["session"]=="T1-active")]
tmp_df = tmp_df.groupby(["sub","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)

rejectSub=[]
for (c,h) in zip(["T1_only","T2_only","T2_only","T1_only"],["hit_t1","hit_t2","fa_t1","fa_t2"]):

    tmp_tmp_df = tmp_df[(tmp_df["condition"]==c)&(df["session"]=="T1-active")]

    z = zscore(tmp_tmp_df[h])
    mask = np.abs(z) > 3
        
    rejectSub.append(tmp_tmp_df[mask]["sub"].values)

rejectSub = np.concatenate(rejectSub)
rejectSub = sorted(set(rejectSub))

print(f"{rejectSub} was rejected because of low accuracy for T1 or T2")

for iSub in rejectSub:
    df = df[df["sub"]!=iSub]
df=df.reset_index(drop=True)
    
reject = [i for i,d in enumerate(datHash["sub"]) if d in rejectSub]

datHash = rejectDat(datHash,reject)

# %%

reject = {}
pp = pre_processing(cfg)

print("pre-processing...")
y,reject["PDR_t1"]  = pp.pre_processing(np.array(datHash["PDR_t1"]).copy())
y2,reject["PDR_t2"] = pp.pre_processing(np.array(datHash["PDR_t2"]).copy())

x = np.linspace(cfg["TIME_START"],cfg["TIME_END"],y.shape[1])
y_bp = np.array(datHash["baseline"])
x = np.linspace(-cfg["WID_BP_ANALYSIS"], 0, y_bp.shape[1])

y_bp = y_bp[:,getNearestValue(x,-1):getNearestValue(x,0)]

#########  rejected BP by velocity
d = np.diff(y_bp)

q75, q25 = np.percentile(d, [75 ,25])
IQR = q75 - q25

lower = q25 - IQR*1.5
upper = q75 + IQR*1.5
d_witout_outliar = d[(d > lower) & (d < upper)]

param = norm.fit(d_witout_outliar)

(a,b) = norm.interval(0.99, loc=param[0], scale=param[1])

ind = np.argwhere(abs(d) > b*3)

reject["BP_diff"] = np.unique(ind[:,0])
reject["BP"] = rejectedByOutlier(datHash, y_bp)
pca,reject["PCA_BP"] = rejectBlink_PCA(y_bp)

tmp_reject=[]
for k in ['BP_diff', 'BP', 'PCA_BP']:
    tmp_reject.append(reject[k])

tmp_reject = list(itertools.chain.from_iterable(tmp_reject))
tmp_reject = sorted(set(tmp_reject))

datHash = rejectDat(datHash, tmp_reject)

df_pupil=[]
for i in np.arange(len(datHash["sub"])):
    df_pupil.append(pd.DataFrame({
        "pupil":datHash["PDR_t2"][i],
        "sub":datHash["sub"][i],
        "session":datHash["session"][i],
        "run":datHash["run"][i],
        "lag":datHash["lag"][i],
        "condition":g[int(datHash["target"][i]-1)],
        "hit_t1":df.iloc[i]["hit_t1"],
        "hit_t2":df.iloc[i]["hit_t2"],
        "P(T2|T1)":df.iloc[i]["P(T2|T1)"],
        "Time":np.arange(len(datHash["PDR_t2"][i]))*(1/cfg["RESAMPLING_RATE"])-3
        }))

df_pupil = pd.concat(df_pupil).reset_index(drop=True)
df_pupil = df_pupil.groupby(["sub","Time","condition","session","lag","P(T2|T1)"],sort=False,as_index=False).agg(np.nanmean)


#%% ------------------ data loading2 -----------------------------------------

# cfg={
# "windowL":[20],
# "TIME_START":-3,
# "TIME_END":3,
# "WID_FILTER":np.array([]),
# "METHOD":1, #subtraction
# "FLAG_LOWPASS":False,
# "WID_BASELINE":[[-0.2,0]],
# "DOT_PITCH":0.33,   
# "VISUAL_DISTANCE":60,
# "acceptMSRange":2,
# "SCREEN_RES":[640*2, 512*2],
# "SCREEN_MM":[432, 324],
# "visualization":False,
# "THRES_DIFF":20
# # }

# f = open(folderName + "/cfg.json")
# cfg.update(json.load(f))
# f.close()


f = open(folderName + "/cfg.json")
cfg=json.load(f)
f.close()

print("data loding...")

datHash = {}

log = sorted(glob.glob(f"./data/**/*_run{normFlg}.msgpack"))[-1].split('/')[2]
log = sorted(glob.glob(f"./data/{log}/*_run{normFlg}.msgpack"))

for r in rejectSub:
    log = [l for l in log if not r in l]

for subName in log:
    with open(subName, "rb") as f:
        tmp_datHash = msgpack.unpackb(f.read(), raw=False)

    for mmName in list(tmp_datHash.keys()):
        if mmName != "cfg":
            if not mmName in list(datHash.keys()):
                datHash[mmName] = []
            datHash[mmName] = datHash[mmName] + tmp_datHash[mmName]
    
#%% ------------------ artifact rejection -----------------------------------

print("pre-processing...")
y_corrected = datHash["PDR"]

# f, amp, psd=getfft_multi(y_corrected,cfg["RESAMPLING_RATE"],1)

# plt.plot(np.array(f).mean(axis=0),np.array(amp).mean(axis=0))
# plt.xlim([0.005,0.5])
# plt.ylim([0,0.06])

# t, power = morlet_cwt_multi(y_corrected, cfg["RESAMPLING_RATE"], np.linspace(0.01, 0.5, 20))

# y_lowpassed = lowpass_filter(y_corrected , 0.1, cfg["RESAMPLING_RATE"])
# y_lowpassed = lowpass_filter(y_corrected , 0.21, cfg["RESAMPLING_RATE"])
y_lowpassed = lowpass_filter(y_corrected , bpass, cfg["RESAMPLING_RATE"])

y = [np.array(y) for y in y_lowpassed]
PCPDevents = getPCPDevents(y, 0.02)

# df_event_BP=[]
# for ev in ["peaks","troughs"]:
#     for iSub,index in tqdm(enumerate(PCPDevents[ev])):
#         for i in index:
            
#             tmp=[np.arange(i+(ii*cfg["RESAMPLING_RATE"]),i+((ii+1)*cfg["RESAMPLING_RATE"])) for ii in np.arange(5)]
            
#             if len(y_lowpassed[iSub][0])>i+(5*cfg["RESAMPLING_RATE"]):
#                 for itmp,tmp_tmp in enumerate(tmp):
#                     df_event_BP.append(pd.DataFrame({
#                     "sub":datHash["sub"][iSub],
#                     # "ind":i,
#                     "BP":np.mean(np.array(y_lowpassed[iSub][0])[tmp_tmp]),
#                     "bin":itmp,
#                     "event":ev
#                     },index=[0]))

# df_event_BP = pd.concat(df_event_BP).reset_index(drop=True)
# df_event_BP = df_event_BP.groupby(["sub","event","bin"],sort=False,as_index=False).agg(np.nanmean)

# plt.figure()
# g = sns.FacetGrid(df_event_BP, 
#                   hue="event",
#                   )
# g.map(sns.pointplot, "bin", "BP")

# plt.figure()
# sns.pointplot(data=df_event_BP,
#               x="event",
#               y="BP",
#               )

# x1=df_event_BP[df_event_BP["event"]=="peaks"]["BP"].to_numpy()
# x2=df_event_BP[df_event_BP["event"]=="troughs"]["BP"].to_numpy()

# t_stat, p_two_sided = ttest_1samp(x2-x1, popmean=0)


# f, amp, psd=getfft_multi(y_lowpassed,cfg["RESAMPLING_RATE"],1)

# y = [np.array(y) for y in y_lowpassed]
# PCPDevents = getPCPDevents(y, 0.02)

# x = np.arange(len(y_corrected[0][0]))*(1/cfg["RESAMPLING_RATE"])

# plt.figure()
# plt.plot(x,np.array(y_corrected[0]).T)
# plt.plot(x,np.array(y_lowpassed[0]).T)
# pcpd = np.array(PCPDevents["peaks"][0])
# # pcpd = np.array(PCPDevents["troughs"][i])
# plt.plot(x[pcpd],np.array(y_lowpassed[0]).T[pcpd],'o')
# plt.xlim([200,400])

# plt.plot(np.array(f).mean(axis=0),np.array(amp).mean(axis=0))
# plt.xlim([0.005,0.5])
# plt.ylim([0,0.06])

# t=[]
# for iTrial in np.arange(len(PCPDevents["troughs"])):
#     for j in np.arange(len(PCPDevents["troughs"][iTrial])-1):
#         t.append((PCPDevents["troughs"][iTrial][j+1] - PCPDevents["troughs"][iTrial][j])*(1/cfg["RESAMPLING_RATE"]))
    
# t=np.array(t)
    
# t2=[]
# for iTrial in datHash["condition_target"]:
#     tmp_t2=[]
#     for j in np.arange(len(iTrial)-1):
#         if iTrial[j][1]==4:
#             tmp_t2.append(iTrial[j][0])

#     tmp_t2_2=[]
#     for j in np.arange(len(tmp_t2)-1):
#         tmp_t2_2.append((tmp_t2[j+1]-tmp_t2[j])*(1/cfg["SAMPLING_RATE"]))

#     t2.append(tmp_t2_2)

# import itertools

# t2 = list(itertools.chain.from_iterable(t2))
# t2=np.array(t2)


# plt.figure()
# sns.histplot(x=t[t<20],
#       kde=True,
#       stat="density",
#       bins=100
#       )

# sns.histplot(x=t2,
#       kde=True,
#       stat="density",
#       bins=100
#       )

# plt.ylim([0,0.3])
# plt.xlim([4,15])

# from scipy.stats import gaussian_kde

# kde1 = gaussian_kde(t[t<20])
# kde2 = gaussian_kde(t2)

# x = np.linspace(0, 20, 2048)

# pdf1 = kde1(x)
# pdf2 = kde2(x)

# plt.figure()

# # plt.plot(x,pdf1)
# # plt.plot(x,pdf2)
# plt.plot(x,pdf1*pdf2)

# %%

iTrial=2

t1_onset=datHash["condition_target"][iTrial]
t1_onset = [round(t[0]/int(cfg["SAMPLING_RATE"]/cfg["RESAMPLING_RATE"])) for t in t1_onset if t[1]==4]

time = np.arange(y[iTrial].shape[1])*(1/cfg["RESAMPLING_RATE"])

plt.figure(figsize=(10,6))

plt.plot(time,y_corrected[iTrial][0],alpha=0.5)
plt.plot(time,y[iTrial].T,'b',alpha=1)
plt.xlabel("Time[s]",fontsize=26)
plt.ylabel("Pupil size[mm]",fontsize=26)

for ev in PCPDevents["troughs"][iTrial]:
    plt.plot(time[np.array(PCPDevents["troughs"][iTrial])],y[iTrial][0,np.array(PCPDevents["troughs"][iTrial])],'bo')
    plt.axvline(x=time[ev], ymin=4, ymax=round(y[iTrial][0,ev],3), color="b",linewidth=0.5)
    
    
    # plt.plot([time[ev], time[ev]], [4, y[iTrial][0,ev]],'b--',linewidth=0.5)

for ev in t1_onset:

    plt.plot([time[ev], time[ev]], [4, y[iTrial][0,ev]],'k--',linewidth=0.5)
    ind=getNearestValue(time[np.array(PCPDevents["troughs"][iTrial])],
                    time[ev])
    
    if (time[ev]>280)&(time[ev]<300):
        tex=round(time[np.array(PCPDevents["troughs"][iTrial])][ind]-time[ev])
        plt.text(time[ev], 4.7, f"Time bin\n{tex}-{tex+1}s")
    
plt.plot(time[np.array(PCPDevents["troughs"][iTrial])],y[iTrial][0,np.array(PCPDevents["troughs"][iTrial])],'bo')

# plt.plot(time[np.array(PCPDevents["peaks"][iTrial])],y[iTrial][0,np.array(PCPDevents["peaks"][iTrial])],'bo')
plt.grid(False)
# plt.xlim([300,400])
plt.xlim([275,305])
plt.ylim([4.5,6.2])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax = plt.gca()

for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("black")
    ax.spines[spine].set_linewidth(1.8)


plt.legend(
    frameon=False,
    fontsize=13,
    loc="upper right"
)
plt.xticks(
    [280,285,290,295,300],
    fontsize=18
)
# yticks
plt.yticks(
    [4.5, 5.0, 5.5, 6.0, 6.5],
    fontsize=18
)

ax = plt.gca()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="x",
    which="major",
    length=30,
    width=1.8,
    direction="out",
    bottom=True,
    top=False
)

ax.tick_params(
    axis="y",
    which="major",
    length=30,
    width=1.8,
    direction="out",
    left=True,
    right=False
)

plt.tight_layout()
# plt.show()

plt.savefig(f"./tmp_fig/fig_{iTrial:02}.pdf")


# %%

mmName = [
    "amp_peaks_st",
    "amp_troughs_st",
    "wave_onset_peaks_st",
    "wave_onset_troughs_st",
    ]

pLen = 10

wave_summary = []
pupil_pktr=[]

plt.figure()
for i,condition in enumerate(datHash["t2"]):
    
    pcpd = PCPDevents["summary"][i].copy()
    # pupil = y[i].reshape(-1)
    pupil = np.array(y_corrected[i][0]).reshape(-1)
    time = np.arange(len(pcpd)) * 1/cfg["RESAMPLING_RATE"]

    tmp_wave_onset_peaks_st = []
    tmp_wave_onset_troughs_st = []

    tmp_amp_peaks_st = []
    tmp_amp_troughs_st = []

    for j in range(len(condition)):
        
        onset_idx = round(condition[j][0]/int(cfg["SAMPLING_RATE"]/cfg["RESAMPLING_RATE"]))
        
        idx_back = onset_idx
        idx_fwd = onset_idx
        
        ## label: peak = 1
        while (pcpd[idx_back]<=0) & (idx_back>0):
            idx_back -= 1

        idx_back_peaks = idx_back
        
        idx_back = onset_idx
        
        ## label: troughs = -1
        while (pcpd[idx_back] >= 0) & (idx_back>0):
            idx_back -= 1
        
        idx_back_troughs = idx_back

        if (len(pupil) > idx_back_troughs+pLen*cfg["RESAMPLING_RATE"])&(len(pupil) > idx_back_peaks+pLen*cfg["RESAMPLING_RATE"]):

            pupil_troughs = pupil[np.arange(idx_back_troughs-pLen*cfg["RESAMPLING_RATE"],
                                            idx_back_troughs+pLen*cfg["RESAMPLING_RATE"])]-pupil[idx_back_troughs]
                                            # idx_back_troughs+pLen*cfg["RESAMPLING_RATE"])]/pupil[idx_back_troughs]*100-100
            pupil_peaks = pupil[np.arange(idx_back_peaks-pLen*cfg["RESAMPLING_RATE"],
                                          idx_back_peaks+pLen*cfg["RESAMPLING_RATE"])]-pupil[idx_back_peaks]
                                          # idx_back_peaks+pLen*cfg["RESAMPLING_RATE"])]/pupil[idx_back_peaks]*100-100
            
            if len(np.argwhere(abs(np.diff(pupil_troughs)) > b*3))==0 & len(np.argwhere(abs(np.diff(pupil_peaks)) > b*3))==0:
                
                pupil_pktr.append(pd.DataFrame({
                    "pupil_troughs":pupil_troughs,
                    "pupil_peaks":pupil_peaks,
                    "sub":datHash["sub"][i],
                    "Session":datHash["session"][i],
                    "Run":datHash["run"][i],
                    "Time":np.arange(pLen*2*cfg["RESAMPLING_RATE"])*(1/cfg["RESAMPLING_RATE"])-pLen
                    }))
            
        if len(np.argwhere(abs(np.diff(pupil_troughs)) > b*3))==0 & len(np.argwhere(abs(np.diff(pupil_peaks)) > b*3))==0:
        
            tmp_amp_peaks_st.append(np.round(pupil[idx_back_peaks] - pupil[onset_idx], 4))
            tmp_amp_troughs_st.append(np.round(pupil[idx_back_troughs] - pupil[onset_idx], 4))
            
            tmp_wave_onset_peaks_st.append(np.round(time[onset_idx] - time[idx_back_peaks], 4))
            tmp_wave_onset_troughs_st.append(np.round(time[onset_idx] - time[idx_back_troughs], 4))
            
        else:

            tmp_amp_peaks_st.append(-1000)
            tmp_amp_troughs_st.append(-1000)

            tmp_wave_onset_peaks_st.append(-1000)
            tmp_wave_onset_troughs_st.append(-1000)

    wave_summary.append(pd.DataFrame({
        "amp_peaks_st":tmp_amp_peaks_st,
        "amp_troughs_st":tmp_amp_troughs_st,
        "wave_onset_peaks_st":tmp_wave_onset_peaks_st,
        "wave_onset_troughs_st":tmp_wave_onset_troughs_st,
        "sub":datHash["sub"][i],
        "run":datHash["run"][i],
        "session":datHash["session"][i],
        }))
    
wave_summary = pd.concat(wave_summary).reset_index(drop=True)

pupil_pktr = pd.concat(pupil_pktr).reset_index(drop=True)

g = sns.FacetGrid(wave_summary[(wave_summary["wave_onset_troughs_st"]>-100)&(wave_summary["wave_onset_troughs_st"]<50)],
                  col_wrap=5,
                  col="sub"
                  )
g.map(sns.histplot, 
      "wave_onset_troughs_st",
      kde=True,
      stat="density",
      bins=10
      )

for ax,colName in zip(g.axes.flat,list(g.axes_dict.keys())):
    ax.axvline(x=10,color="k")



# %% 

df[mmName]=0
for iSub in df["sub"].unique():
    for iRun in df["run"].unique():
        for iSession in df["session"].unique():
            for m in mmName:
                df.loc[(df["sub"]==iSub)&
                       (df["session"]==iSession)&
                       ((df["condition"]=="T2_only")|(df["condition"]=="Both"))&
                       (df["run"]==iRun),m] = wave_summary[(wave_summary["sub"]==iSub)&
                                                           (wave_summary["session"]==iSession)&
                                                           (wave_summary["run"]==iRun)][m].values

df = df.drop(tmp_reject)

# # %%

# plt.figure()
# g = sns.FacetGrid(df, 
#                   col="sub",
#                   row="session",
#                   # hue="lag",
#                   )
# # g.map(sns.pointplot, "time_bin", "y_resid_smooth_norm",
# g.map(sns.scatterplot, "wave_onset", "P(T2|T1)_smooth_norm",
#       # errorbar=None
#       # errorbar="ci"
#       # errorbar="se"
#       )
# g.set(xlim=[-1,1])
# g.set(ylim=[-1,1])

# %% regressout run effect

df_tmp = df[(df["condition"]=="T2_only")|(df["condition"]=="Both")]
df_tmp["y"] = df_tmp["hit_t2"]

df_tmp[["y_resid","RT_t2_resid"]] = -100
df[["y_resid","RT_t2_resid","y_resid_smooth_norm","P(T2|T1)_smooth_norm","RT_t2_resid_norm","time_bin"]] = -100

for iSession in df_tmp["session"].unique():
    for iTarget in df_tmp["condition"].unique():
        
        df_tmp_tmp = df_tmp[(df_tmp["session"]==iSession)&
                            (df_tmp["condition"]==iTarget)]

        model = smf.mixedlm("y ~ run", df_tmp_tmp, groups=df_tmp_tmp["sub"])
        result = model.fit()
    
        df_tmp.loc[(df_tmp["session"]==iSession)&
                   (df_tmp["condition"]==iTarget),"y_resid"] = result.resid + result.params["Intercept"]

        df.loc[(df["session"]==iSession)&
               (df["condition"]==iTarget),"y_resid"] = result.resid + result.params["Intercept"]
        

for iSession in df_tmp["session"].unique():
 
        df_tmp_tmp = df_tmp[(df_tmp["session"]==iSession)&
                            (df_tmp["condition"]==iTarget)&
                            (df_tmp["RT_t2"]!=-1)]

        model = smf.mixedlm("RT_t2 ~ run", df_tmp_tmp, groups=df_tmp_tmp["sub"])
        result = model.fit()

        df.loc[(df["session"]==iSession)&
               (df["condition"]==iTarget)&
               (df["RT_t2"]!=-1),"RT_t2_resid"] = result.resid + result.params["Intercept"]
        
        df_tmp.loc[(df_tmp["session"]==iSession)&
                   (df_tmp["condition"]==iTarget)&
                   (df_tmp["RT_t2"]!=-1),"RT_t2_resid"] = result.resid + result.params["Intercept"]


# %%

sTime = 0
eTime = 11
jitter=1

target = "_pupil"
# event = "_peaks"
event = "_troughs"

df_tmp["wave_onset"] = df_tmp[f"wave_onset{event}_st"]
df["wave_onset"] = df[f"wave_onset{event}_st"]

# df_tmp["wave_onset"] = df_tmp[f"amp{event}_st"]
# df["wave_onset"] = df[f"amp{event}_st"]


df_tmp = df_tmp[(df_tmp["wave_onset"]<=eTime)]

df_tmp_tmp = df_tmp.copy()
df_tmp_tmp["P(T2|T1)_smooth_norm"] = -1

# jitter=0.25
# jitter=0.5

df_ave=[]
df_tmp_tmp["time_bin"]=-1

for iSub in df_tmp_tmp["sub"].unique():
    for iSession in df_tmp_tmp["session"].unique():
        for iLag in df_tmp_tmp["lag"].unique():

            ind = (df_tmp_tmp["sub"]==iSub)&(df_tmp_tmp["session"]==iSession)&(df_tmp_tmp["lag"]==iLag)

            t = df_tmp_tmp[ind]["y_resid"].values
            # if np.isnan(t[0] / t.mean()) ==True:
            #     df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = 0
            # else:
                # df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = t / t.mean() * 100 -100

            df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = t - t.mean()
            
            t = df_tmp_tmp[ind]["P(T2|T1)"].values
            df_tmp_tmp.loc[ind,"P(T2|T1)_smooth_norm"] = t - t.mean()
            
            t = df_tmp_tmp[ind]["RT_t2_resid"].values
            df_tmp_tmp.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t)
            
            for iTarget in df_tmp_tmp["condition"].unique():
                
                ind = (df["sub"]==iSub)&(df["session"]==iSession)&(df["lag"]==iLag)&(df["wave_onset"]<=eTime)&(df["condition"]==iTarget)
                       
                t = df[ind]["y_resid"].values
                df.loc[ind,"y_resid_smooth_norm"] = t - t.mean()
                # if np.isnan(t[0] / t.mean()) ==True:
                #     df.loc[ind,"y_resid_smooth_norm"] = 0          
                # else:
                #     df.loc[ind,"y_resid_smooth_norm"] = t / t.mean() * 100 -100
                #     print(f"{iSession},{iLag},{iTarget}")
                #     print(t / t.mean() * 100 - 100)
                            
                t = df[ind]["P(T2|T1)"].values
                df.loc[ind,"P(T2|T1)_smooth_norm"] = t - t.mean()

                t = df[ind]["RT_t2_resid"].values            
                df.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t)
                # df.loc[ind,"RT_t2_resid_norm"] = t / np.nanmean(t)* 100
            

for iTime in np.arange(sTime,eTime-jitter,jitter):

    t = df_tmp_tmp[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter)].copy()
    df_tmp_tmp.loc[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter),"time_bin"]=iTime
    
    df.loc[(df["wave_onset"]>iTime)&(df["wave_onset"]<iTime+jitter),"time_bin"] = iTime
    
    del t["condition"]
    t = t.groupby(["sub","session","lag"],sort=False,as_index=False).agg(np.nanmean)

    # t = t.groupby(["sub","lag"],sort=False,as_index=False).agg(np.nanmean)
    t["time_bin"] = iTime
    df_ave.append(t)
    
df_ave = pd.concat(df_ave).reset_index(drop=True)

# %%

plt.figure()
g = sns.FacetGrid(df_ave, 
                  # col="lag",
                  row="session",
                  # hue="lag",
                  )
g.map(sns.pointplot, "time_bin", "y_resid_smooth_norm",
# g.map(sns.pointplot, "time_bin", "P(T2|T1)_smooth_norm",
      # errorbar=None
      errorbar="ci"
      # errorbar="se"
      ).add_legend()    
g.fig.set_figheight(8)
g.fig.set_figwidth(4)
# g.fig.set_figwidth(8)
# plt.axhline(y=0,color="k")
for ax,colName in zip(g.axes.flat,list(g.axes_dict.keys())):
    ax.axhline(y=0,color="k")

# %%

tmp_df_ave = df_ave.copy()
tmp_df_ave = tmp_df_ave[tmp_df_ave["time_bin"]>=0]

# del tmp_["condition"]
# tmp_ = tmp_.groupby(["sub","session","time_bin"],sort=False,as_index=False,numeric_only=True).agg(np.nanmean)
tmp_df_ave = tmp_df_ave.groupby(
    ["sub","session","time_bin"],
    sort=False,
    as_index=False
).mean(numeric_only=True)

for iSession in tmp_df_ave["session"].unique():

    df_tmp = tmp_df_ave[(tmp_df_ave["session"]==iSession)]

    df_pval=[]
    for iTime in sorted(tmp_df_ave["time_bin"].unique()):

        t = df_tmp[
            # (df_tmp["lag"]==iLag)&
            (df_tmp["time_bin"]==iTime)
            ]

        t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=0, alternative="greater")

        # t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=1)
        # t_stat, p_two_sided = ttest_1samp(t["P(T2|T1)_smooth_norm"], popmean=0)

        df_pval.append(pd.DataFrame({
            # "lag":iLag,
            "time":iTime,
            "tval":t_stat,
            "pval":round(p_two_sided,4),
            },index=[0]))
            
    print(f"{iSession}:{df_pval}")
    
    # df_pval = pd.concat(df_pval)
    
    
# %%

for iSession in df_ave["session"].unique():
    
    df_tmp = df_ave[(df_ave["session"]==iSession)]

    df_pval=[]
    for iLag in df_tmp["lag"].unique():
        for iTime in df_tmp["time_bin"].unique():
    
            t = df_tmp[(df_tmp["lag"]==iLag)&
                       (df_tmp["time_bin"]==iTime)]
    
            t["y_resid_smooth_norm"].mean()
            t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=0)
            # t_stat, p_two_sided = ttest_1samp(t["y_resid_smooth_norm"], popmean=1)
            # t_stat, p_two_sided = ttest_1samp(t["P(T2|T1)_smooth_norm"], popmean=0)
    
            df_pval.append(pd.DataFrame({
                "lag":iLag,
                "time":iTime,
                "tval":t_stat,
                "pval":round(p_two_sided,4),
                },index=[0]))
            
    df_pval = pd.concat(df_pval)
    
    df_pval['annot'] = df_pval['pval'].apply(lambda p: '*' if p < 0.05 else '')
    # annot_table = df_pval.pivot(index='lag', columns='time', values='pval')
    annot_table = df_pval.pivot(index='lag', columns='time', values='annot')
    
    
    pivot_table = df_tmp.pivot_table(
        index='lag',
        columns='time_bin',
        values='y_resid_smooth_norm'
        # values='P(T2|T1)_smooth_norm'
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
    ax.set_xlabel("Time[s]")
    ax.set_ylabel("Lag")
    ax.invert_yaxis()

# %% classify based on pupil

def groupwise_qcut(x):
    try:
        return pd.qcut(x, q=5, labels=False, duplicates='drop')
    except ValueError:
        return pd.Series([np.nan] * len(x), index=x.index)

df['PD_bin'] = (
    df.groupby(['sub', 'session', 'lag','condition'])['PD']
    .transform(groupwise_qcut)
)

df_tertile_origin = df[(df["condition"] == "Both")].copy()
df_tertile = pd.DataFrame()

df["pupilSizeClass"]=""

# bp_tag = ["Small","Large"]
for iSub in np.unique(df_tertile_origin["sub"]):
    for iLag in np.unique(df_tertile_origin["lag"]):
        for iSession in np.unique(df_tertile_origin["session"]):
            
            tmp = df_tertile_origin[(df_tertile_origin["sub"] == iSub) & 
                                    (df_tertile_origin["lag"] == iLag) &
                                    (df_tertile_origin["session"] == iSession)
                                    ]
            tmp = tmp.sort_values("PD")
            
            ind = (df["sub"] == iSub) & (df["lag"] == iLag) & (df["session"] == iSession)&(df["condition"] == "Both")

            t = df[ind]["PD"].mean()
            
            df.loc[ind&(df["PD"] < t),"pupilSizeClass"]="Small"
            df.loc[ind&(df["PD"] > t),"pupilSizeClass"]="Large"


df_sum = df.groupby(["sub","pupilSizeClass","session","lag","condition"],as_index=False).agg(np.sum)
df_sum = df_sum[df_sum["condition"]=="Both"]

t=[]
for index, row in df_sum.iterrows():
    n = row["hit_t2"] + row["miss_t2"]

    t.append(SDT(row['hit_t2']/n, 0))
t = pd.concat(t)

for mmName in list(t.keys()):
    df_sum[mmName] = t[mmName].values

# %%# P(T2|P1) based on pupil size

df_tmp = df[df["condition"]=="Both"].reset_index(drop=True)

df_tmp = df_tmp.groupby(["sub","session",'P(T2|T1)'], sort=False, as_index=False).agg("mean",numeric_only=True)
df_tmp["p"]=df_tmp["P(T2|T1)"]

plt.figure(figsize=(7,7))
grid = sns.FacetGrid(df_tmp, col="session")
# grid.map(sns.pointplot, "lag", "res_t2", dodge=True)
grid.map(sns.pointplot, "P(T2|T1)", "PD", dodge=True)
plt.legend()
plt.ylabel("Accuracy(P(T2))")
plt.xlabel("Lag")

print(AnovaRM(data=df_tmp, 
              depvar='PD',
              subject='sub', 
              within=['p','session']).fit())

# %%

df_tmp = df[df["condition"]=="Both"].reset_index(drop=True)

del df_tmp["pupilSizeClass"]
df_tmp = df_tmp.groupby(["sub","session",'PD_bin'], sort=False, as_index=False).agg("mean",numeric_only=True)

plt.figure(figsize=(7,7))
grid = sns.FacetGrid(df_tmp, col="session")
grid.map(sns.pointplot, "PD_bin","P(T2|T1)", dodge=True)
plt.legend()
plt.ylabel("Accuracy(P(T2))")
plt.xlabel("Lag")


# %%

# x = np.arange(tmp_pupil1[0].shape[1])*(1/cfg["RESAMPLING_RATE"])-pLen

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2,1, height_ratios=[1, 2], hspace=0.05, wspace=0.05)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

# ax0.plot(x,tmp_pupil2.mean(axis=0))
# ax0.set_ylabel('Pupil size [mm]')
# ax0.set_xlim([0,pLen])
# ax.axhline(y=0,color="k")

pupil_pktr = pupil_pktr.groupby(["sub","Time","Session"], sort=False, as_index=False).agg(np.nanmean)

sns.lineplot(
    data=pupil_pktr,
    x="Time",
    y="pupil_troughs",
    errorbar=None,
    legend=False,
    ax=ax0
)
ax0.set_xlim([0,pLen])

# for ax, lag_val in zip(g.axes.flat, ['lag'].unique()):

tmp_df_ave = df_ave
# tmp_ = [np.isnan(["RT_t2_resid_norm"])==False]

sns.pointplot(
    data=tmp_df_ave,
    x="time_bin",
    y="P(T2|T1)_smooth_norm",
    # y="y_resid_smooth_norm",
    # hue="sub",
    # y="RT_t2_resid_norm",
    # errorbar=None,
    errorbar="ci",
    legend=False,
    # color="C0",
    ax=ax1
)

ax1.axhline(y=0,color="k")

def damped_sinusoid(t, A, lambda_, omega, phi):
    return A * np.exp(-lambda_ * t) * np.sin(omega * t + phi)

# y = tmp_[tmp_["session"]=="T1-active"]["y_resid_smooth_norm"].values
# t = tmp_[tmp_["session"]=="T1-active"]["wave_onset"].values

y = tmp_df_ave["y_resid_smooth_norm"].values
t = tmp_df_ave["wave_onset"].values

from scipy.optimize import differential_evolution

# A, lambda, omega, phi
bounds = [(0.001, 0.2), (0.01, 4), (0.5, 2), (-np.pi, np.pi)]

def loss(params):
    return np.sum((damped_sinusoid(t, *params) - y)**2)

result = differential_evolution(loss, bounds)
params = result.x

t_fit = np.linspace(min(t), max(t), 100)
y_fit = damped_sinusoid(t_fit, *params)
plt.plot(t_fit, y_fit, label="Fitted Curve", linestyle='--')

# %%

# def damped_sinusoid(t, A, lambda_, omega, phi):
#     return A * np.exp(-lambda_ * t) * np.sin(omega * t + phi)

# # A, lambda, omega, phi
# bounds = [(0.001, 0.2), (0.01, 4), (0.5, 2), (-np.pi, np.pi)]

# def fit_one_subject(t, y, bounds):
#     m = np.isfinite(t) & np.isfinite(y)
#     t_ = np.asarray(t)[m]
#     y_ = np.asarray(y)[m]

#     if len(t_) < 8 or np.nanstd(y_) < 1e-8:
#         return None

#     def loss(params):
#         return np.sum((damped_sinusoid(t_, *params) - y_)**2)

#     result = differential_evolution(loss, bounds, seed=0, polish=True)
#     return result.x  # params

# tmp_df = tmp_.copy()
# subs = tmp_df["sub"].unique()

# t_min = tmp_df["wave_onset"].min()
# t_max = tmp_df["wave_onset"].max()
# t_common = np.linspace(t_min, t_max, 200)

# params_rows = []
# pred_rows = []

# for sub in subs:
#     d = tmp_df[tmp_df["sub"] == sub]
#     t = d["wave_onset"].values
#     y = d["y_resid_smooth_norm"].values

#     params = fit_one_subject(t, y, bounds)
#     if params is None:
#         continue

#     y_pred = damped_sinusoid(t_common, *params)

#     params_rows.append({
#         "sub": sub,
#         "A": params[0],
#         "lambda": params[1],
#         "omega": params[2],
#         "phi": params[3],
#     })

#     pred_rows.append(pd.DataFrame({
#         "sub": sub,
#         "t": t_common,
#         "y_fit": y_pred
#     }))

# params_df = pd.DataFrame(params_rows)
# pred_df = pd.concat(pred_rows, ignore_index=True)

# from scipy.stats import ttest_1samp

# alpha = 0.05
# stat_df = (
#     pred_df
#     .groupby("t")
#     .apply(lambda d: ttest_1samp(d["y_fit"], popmean=0, nan_policy="omit"))
#     .reset_index()
# )

# stat_df["tval"] = stat_df[0].apply(lambda x: x.statistic)
# stat_df["pval"] = stat_df[0].apply(lambda x: x.pvalue)

# stat_df["sig"] = stat_df["pval"] < alpha

# from statsmodels.stats.multitest import fdrcorrection

# # stat_df["p_fdr"] = fdrcorrection(stat_df["pval"])[1]
# # stat_df["sig"] = stat_df["p_fdr"] < 0.05

# sig_segments = []
# in_segment = False

# for i, row in stat_df.iterrows():
#     if row["sig"] and not in_segment:
#         start_t = row["t"]
#         in_segment = True
#     elif not row["sig"] and in_segment:
#         end_t = stat_df.loc[i-1, "t"]
#         sig_segments.append((start_t, end_t))
#         in_segment = False

# if in_segment:
#     sig_segments.append((start_t, stat_df.iloc[-1]["t"]))

# mean_fit = pred_df.groupby("t", as_index=False)["y_fit"].mean()
# sem_fit = pred_df.groupby("t")["y_fit"].sem().reset_index(name="y_fit_sem")

# # ---- 描画 ----
# # plt.figure(figsize=(6,4))
# # # 被験者平均フィット
# # plt.plot(mean_fit["t"], mean_fit["y_fit"], linestyle="--", label="Mean fitted curve")
# # # SEM帯
# # plt.fill_between(sem_fit["t"],
# #                  mean_fit["y_fit"].values - sem_fit["y_fit_sem"].values,
# #                  mean_fit["y_fit"].values + sem_fit["y_fit_sem"].values,
# #                  alpha=0.2, label="SEM")

# # plt.axhline(0, color="k", linewidth=1)
# # plt.legend()
# # plt.xlabel("wave_onset")
# # plt.ylabel("y_resid_smooth_norm (fit)")
# # plt.tight_layout()
# # plt.show()

# # パラメータ平均も見たいなら
# ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

# sns.pointplot(
#     data=tmp_,
#     x="time_bin",
#     y="P(T2|T1)_smooth_norm",
#     errorbar="ci",
#     legend=False,
#     ax=ax1
# )

# ax1.axhline(y=0, color="k")

# ymin, ymax = ax1.get_ylim()
# y_bar = ymin + 0.3 * (ymax - ymin)
# y_star = ymin + 0.08 * (ymax - ymin)

# for t0, t1 in sig_segments:
#     ax1.plot([t0, t1], [y_bar, y_bar], color="red", linewidth=4)
#     # ax1.text((t0 + t1) / 2, y_star, "*",
#     #          ha="center", va="center", fontsize=14, color="red")
    
# # print(params_df.describe())
    

# %%

# df_event_BP.to_json(f"{savefolder}/df_event_BP{normFlg}.json")

df.to_json(f"{savefolder}/df{normFlg}.json")
df_ave.to_json(f"{savefolder}/df2{normFlg}.json")
pupil_pktr.to_json(f"{savefolder}/df_pupilTimeCourse{normFlg}.json")
df_sum.to_json(f"{savefolder}/df_sum{normFlg}.json")

df_pupil = df_pupil[(df_pupil["condition"]=="T2_only")|(df_pupil["condition"]=="Both")]
df_pupil.to_json(f"{savefolder}/df_pupil{normFlg}.json")


# df_event_BP.to_json(f"{folderName}/df_event_BP{normFlg}_{str(bpass).replace('.','')}.json")
# df.to_json(f"{folderName}/df{normFlg}_{str(bpass).replace('.','')}.json")
# .to_json(f"{folderName}/df2{normFlg}_{str(bpass).replace('.','')}.json")
# pupil_pktr.to_json(f"{folderName}/df_pupilTimeCourse{normFlg}_{str(bpass).replace('.','')}.json")
# df_sum.to_json(f"{folderName}/df_sum{normFlg}_{str(bpass).replace('.','')}.json")

# df_pupil = df_pupil[(df_pupil["condition"]=="T2_only")|(df_pupil["condition"]=="Both")]
# df_pupil.to_json(f"{folderName}/df_pupil{normFlg}_{str(bpass).replace('.','')}.json")

# plt.figure()
# g = sns.FacetGrid(, 
#                   # col="lag",
#                   # row="session",
#                   hue="lag",
#                   )
# g.map(sns.pointplot, "time", "P(T2|T1)_smooth_norm",
#       errorbar=None,
#       ax=ax1
#       # errorbar="se"
#       ).add_legend()
# g.fig.set_figheight(8)
# # g.fig.set_figwidth(20)
# g.fig.set_figwidth(8)
# plt.axhline(y=0,color="k")

# ax1.set_xlabel('Time')
# ax1.set_ylabel('Parcels sorted by PG values')
# ax1.grid(False)

# %%

df_tmp = df[(df["condition"]=="T2_only")|(df["condition"]=="Both")].reset_index(drop=True)
df_tmp = df[df["time_bin"] >=0].reset_index(drop=True)

df_tmp["count"]=-100

for iSub in np.unique(df_tmp["sub"]):
    for iSession in np.unique(df_tmp["session"]):
        for iTime in np.unique(df_tmp["time_bin"]):
            
            t = df_tmp[(df_tmp["sub"]==iSub)&
                       (df_tmp["session"]==iSession)&
                       (df_tmp["time_bin"]==iTime)]
    
            df_tmp.loc[(df_tmp["sub"]==iSub)&
                       (df_tmp["session"]==iSession)&
                       (df_tmp["time_bin"]==iTime),"count"]=len(t)
    
del df_tmp["pupilSizeClass"]
df_tmp = df_tmp.groupby(["sub","time_bin","session"],sort=False,as_index=False).agg("mean",numeric_only=True)

# %%

plt.figure()
sns.scatterplot(data=df_tmp,
                x="count", 
                hue="session",
                y="y_resid_smooth_norm")

plt.figure(figsize=(10,4))
sns.pointplot(data=df_tmp,
                x="count", 
                hue="session",
                y="y_resid_smooth_norm")


plt.figure()
sns.pointplot(data=df_tmp,
              x="time_bin", 
              y="count",
              hue="session",
              errorbar="ci")

# %%

eTime=7
tmp = wave_summary[(wave_summary["wave_onset_troughs_st"]>=sTime)&
                   (wave_summary["wave_onset_troughs_st"]<=eTime)]
tmp["count"]=1

jitter=1
for iTime in np.arange(eTime):

    t = tmp[(tmp["wave_onset_troughs_st"]>iTime)&(tmp["wave_onset_troughs_st"]<iTime+jitter)].copy()
    tmp.loc[(tmp["wave_onset_troughs_st"]>iTime)&(tmp["wave_onset_troughs_st"]<iTime+jitter),"time_bin"]=f"{iTime}-{iTime+1}"
    
tmp_df = tmp.groupby(["sub","session","time_bin"],sort=False,as_index=False).agg("sum",numeric_only=True)


g = sns.FacetGrid(tmp_df,
                  hue="session")

g.map(sns.pointplot, 
      "time_bin",
      "count",
      order=np.sort(tmp_df["time_bin"].unique())
      ).add_legend()
g.fig.set_figheight(6)
g.fig.set_figwidth(8)

# if __name__ == '__main__':
 
#     # for bpass in np.round(np.arange(0.1,0.31,0.05),2):
#     #     run(bpass)

#     run(0.15)