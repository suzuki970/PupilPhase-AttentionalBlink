#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
## run from this folder, also when executed from the workspace root (e.g. Positron)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath("Exp1" if os.path.isdir("Exp1") else ".")
os.chdir(HERE)
sys.path.append(os.path.join(HERE, "..", "lib"))

import glob
import itertools

import json
import warnings

import msgpack
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from band_pass_filter import nan_lowpass
from pre_processing_cls import (
    getNearestValue,
    getPCPDevents,
    pre_processing,
    rejectDat,
    rejectedByOutlier,
)
from rejectBlink_PCA import rejectBlink_PCA
from scipy.stats import chi2, norm, ttest_1samp, zscore
from statsmodels.stats.multitest import multipletests


warnings.simplefilter("ignore")

bpass=0.15

#% ------------------ initial settings ------------------

cfg={
"windowL":0,
"TIME_START":-3,
"TIME_END":3,
"WID_FILTER":np.array([]),
"THRES_DIFF":10,
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


folderName = sorted(glob.glob(os.path.join("./data/*")))[-1]

savefolder = folderName

f = open(folderName + "/cfg.json")
cfg.update(json.load(f))
f.close()

print("data loding...")

log = sorted(glob.glob(f"./data/**/*_trial.msgpack"))[-1].split('/')[2]
log = sorted(glob.glob(f"./data/{log}/*_trial.msgpack"))

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
for mmName in ["sub","session","lag","run","time"]:
    df[mmName] = datHash[mmName]

g = ["no_sound","T1_only","T2_only","Both"]
df["condition"] = [g[int(t-1)] for t in datHash["target"]]

y_bp = np.array(datHash["baseline"])
x = np.linspace(-cfg["WID_BP_ANALYSIS"], 0, y_bp.shape[1])

y_bp = y_bp[:,getNearestValue(x,-1):getNearestValue(x,0)]

df["PD"] = np.nanmean(y_bp, axis=1)
df["EventPD_t1"] = np.nanmean(np.array(datHash["PDR_t1"])[:,getNearestValue(time_x,-1):getNearestValue(time_x,0)], axis=1)
df["EventPD_t2"] = np.nanmean(np.array(datHash["PDR_t2"])[:,getNearestValue(time_x,-1):getNearestValue(time_x,0)], axis=1)

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

mask = (df["condition"].isin(["T1_only", "Both"]))&(df["session"]=="T1-attend")
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
df["P(T2|T1)"][(df["hit_t1"]==1)&(df["hit_t2"]==1)&(df["session"]=="T1-attend")]=1
df["P(T2|T1)"][(df["hit_t2"]==1)&(df["session"]=="T1-ignore")]=1

#%% reject sub

tmp_df = df[(df["session"]=="T1-attend")]
tmp_df = tmp_df.groupby(["sub","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)

rejectSub=[]
for (c,h) in zip(["T1_only","T2_only","T2_only","T1_only"],["hit_t1","hit_t2","fa_t1","fa_t2"]):

    tmp_tmp_df = tmp_df[(tmp_df["condition"]==c)&(df["session"]=="T1-attend")]

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


reject = {}
pp = pre_processing(cfg)

print("pre-processing...")
y,reject["PDR_t1"]  = pp.pre_processing(np.array(datHash["PDR_t1"]).copy())
_,reject["PDR_t2"] = pp.pre_processing(np.array(datHash["PDR_t2"]).copy())

y_bp = np.array(datHash["baseline"])
x = np.linspace(-cfg["WID_BP_ANALYSIS"], 0, y_bp.shape[1])

y_bp = y_bp[:,getNearestValue(x,-1):getNearestValue(x,0)]

THRES_BP_DIFF = None
d = np.diff(y_bp)


q75, q25 = np.nanpercentile(d, [75 ,25])     # NaN diffs (gaps) are ignored
IQR = q75 - q25

lower = q25 - IQR*1.5
upper = q75 + IQR*1.5
d_witout_outliar = d[(d > lower) & (d < upper)]

param = norm.fit(d_witout_outliar)

(a,b) = norm.interval(0.99, loc=param[0], scale=param[1])

thres_bp_diff = b*3 if THRES_BP_DIFF is None else THRES_BP_DIFF

ind = np.argwhere(abs(d) > thres_bp_diff)


reject["BP_diff"] = np.unique(ind[:,0])
reject["BP"] = rejectedByOutlier(datHash, np.nanmean(y_bp, axis=1))
pca,reject["PCA_BP"] = rejectBlink_PCA(y_bp)

tmp_reject=[]
for k in ['BP_diff', 'BP', 'PCA_BP']:
    tmp_reject.append(reject[k])

tmp_reject = list(itertools.chain.from_iterable(tmp_reject))
tmp_reject = sorted(set(tmp_reject))

rejected_windows = {}
for k in tmp_reject:
    onset = datHash["seq_onset"][k] / (cfg["SAMPLING_RATE"]/cfg["RESAMPLING_RATE"])
    rejected_windows.setdefault((datHash["sub"][k], datHash["session"][k], datHash["run"][k]), []).append(
        (onset - 1*cfg["RESAMPLING_RATE"], onset))

datHash = rejectDat(datHash, tmp_reject)

df_kept = df.drop(index=tmp_reject).reset_index(drop=True)
assert len(df_kept) == len(datHash["sub"]) and (df_kept["sub"].to_numpy() == np.array(datHash["sub"])).all()

df_pupil=[]
for i in np.arange(len(datHash["sub"])):
    df_pupil.append(pd.DataFrame({
        "pupil":datHash["PDR_t2"][i],
        "pupil_t1":datHash["PDR_t1"][i],
        "sub":datHash["sub"][i],
        "session":datHash["session"][i],
        "run":datHash["run"][i],
        "lag":datHash["lag"][i],
        "condition":g[int(datHash["target"][i]-1)],
        "hit_t1":df_kept.iloc[i]["hit_t1"],
        "hit_t2":df_kept.iloc[i]["hit_t2"],
        "P(T2|T1)":df_kept.iloc[i]["P(T2|T1)"],
        "Time":np.arange(len(datHash["PDR_t2"][i]))*(1/cfg["RESAMPLING_RATE"])-3
        }))

df_pupil = pd.concat(df_pupil).reset_index(drop=True)
df_pupil = df_pupil.groupby(["sub","Time","condition","session","lag","P(T2|T1)"],sort=False,as_index=False).agg(np.nanmean)

#%% ------------------ data loading -----------------------------------------

f = open(folderName + "/cfg.json")
cfg=json.load(f)
f.close()

print("data loding...")

datHash = {}

log = sorted(glob.glob(f"./data/**/*_run.msgpack"))[-1].split('/')[2]
log = sorted(glob.glob(f"./data/{log}/*_run.msgpack"))

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

#%% fraction of interpolated samples (zeroArray) in the included participants
zero_frac = pd.DataFrame({
    "sub": datHash["sub"],
    "zero_frac": [np.mean(np.asarray(z).reshape(-1)[np.isfinite(np.asarray(p, dtype=float).reshape(-1))])
                  for z, p in zip(datHash["zeroArray"], datHash["PDR"])],
}).groupby("sub")["zero_frac"].mean()
# summary for the "Pupil data quality" paragraph of results.Rmd
pd.DataFrame({"n_sub": [len(zero_frac)], "interp_frac_mean": [zero_frac.mean()],
              "interp_frac_std": [zero_frac.std()]}).to_csv(f"{savefolder}/zero_frac_summary.csv", index=False)

#%% ------------------ artifact rejection -----------------------------------

print("pre-processing...")
y_corrected = datHash["PDR"]

def detect_events(cutoff):
    """Low-pass filter (cutoff Hz) and pupil peak / trough detection; samples in NaN gaps are not used"""
    y_filled, y_missing = nan_lowpass(y_corrected, cutoff, cfg["RESAMPLING_RATE"])
    y = [np.where(m, np.nan, f) for f, m in zip(y_filled, y_missing)]   # low-passed, gaps NaN


    PCPDevents = getPCPDevents(y_filled, 0.02)

    for i, m in enumerate(y_missing):
        for ev in ["peaks", "troughs"]:
            drop = [k for k in PCPDevents[ev][i] if m[k]]
            PCPDevents[ev][i] = [k for k in PCPDevents[ev][i] if not m[k]]
            PCPDevents["summary"][i] = list(PCPDevents["summary"][i])
            for k in drop:
                PCPDevents["summary"][i][k] = 0
    return y, PCPDevents


y, PCPDevents = detect_events(bpass)

iTrial=2

t1_onset=datHash["condition_target"][iTrial]
t1_onset = [round(t[0]/int(cfg["SAMPLING_RATE"]/cfg["RESAMPLING_RATE"])) for t in t1_onset if t[1]==4]

time = np.arange(y[iTrial].shape[1])*(1/cfg["RESAMPLING_RATE"])


#%% example trace for Figure 3A (drawn in figure.Rmd)
troughs = np.array(PCPDevents["troughs"][iTrial])
df_example_trace = pd.DataFrame({
    "sub": datHash["sub"][iTrial],
    "session": datHash["session"][iTrial],
    "time": time,
    "pupil": np.array(y_corrected[iTrial]).reshape(-1),
    "pupil_lp": y[iTrial].reshape(-1),
})
df_example_events = pd.concat([
    pd.DataFrame({"event": "trough", "time": time[troughs], "pupil_lp": y[iTrial][0, troughs],
                  "time_from_trough": 0.0}),
    pd.DataFrame({"event": "sequence", "time": time[t1_onset], "pupil_lp": y[iTrial][0, t1_onset],
                  "time_from_trough": [time[ev] - time[troughs[troughs <= ev][-1]] if np.any(troughs <= ev) else np.nan
                                       for ev in t1_onset]}),
]).reset_index(drop=True)
df_example_trace.to_json(f"{savefolder}/df_example_trace.json")
df_example_events.to_json(f"{savefolder}/df_example_events.json")


def phase_analysis(df, PCPDevents):
    """Phase of the slow pupil fluctuation at T2 onset (time since the preceding trough / peak)
    and run-corrected T2|T1 accuracy per participant x session x lag x 1-s time bin"""
    df = df.copy()
    mmName = [
        "amp_peaks_st",
        "amp_troughs_st",
        "wave_onset_peaks_st",
        "wave_onset_troughs_st",
        ]

    pLen = 7

    MAX_INTERP_SHARE = 0.8

    wave_summary = []
    pupil_pktr=[]

    for i,condition in enumerate(datHash["t2"]):

        pcpd = PCPDevents["summary"][i].copy()
        pupil = np.array(y_corrected[i][0]).reshape(-1)
        zeroArray = np.array(datHash["zeroArray"][i]).reshape(-1)
    
        tmp_wave_onset_peaks_st = []
        tmp_wave_onset_troughs_st = []

        tmp_amp_peaks_st = []
        tmp_amp_troughs_st = []

        for j in range(len(condition)):
        
            onset_idx = round(condition[j][0]/int(cfg["SAMPLING_RATE"]/cfg["RESAMPLING_RATE"]))
        
            idx_back = onset_idx
        
            while (pcpd[idx_back]<=0) & (idx_back>0):
                idx_back -= 1

            idx_back_peaks = idx_back
        
            idx_back = onset_idx
        
            while (pcpd[idx_back] >= 0) & (idx_back>0):
                idx_back -= 1
        
            idx_back_troughs = idx_back

            keep = True
            if (len(pupil) > idx_back_troughs+pLen*cfg["RESAMPLING_RATE"])&(len(pupil) > idx_back_peaks+pLen*cfg["RESAMPLING_RATE"]):

                pupil_troughs = pupil[np.arange(idx_back_troughs-pLen*cfg["RESAMPLING_RATE"],
                                                idx_back_troughs+pLen*cfg["RESAMPLING_RATE"])]-pupil[idx_back_troughs]
                pupil_peaks = pupil[np.arange(idx_back_peaks-pLen*cfg["RESAMPLING_RATE"],
                                              idx_back_peaks+pLen*cfg["RESAMPLING_RATE"])]-pupil[idx_back_peaks]

                # exclude trials whose trough or peak window is mostly interpolated
                interp_troughs = zeroArray[np.arange(idx_back_troughs-pLen*cfg["RESAMPLING_RATE"],
                                                     idx_back_troughs+pLen*cfg["RESAMPLING_RATE"])].mean()
                interp_peaks = zeroArray[np.arange(idx_back_peaks-pLen*cfg["RESAMPLING_RATE"],
                                                   idx_back_peaks+pLen*cfg["RESAMPLING_RATE"])].mean()
            
                keep = keep and (interp_troughs < MAX_INTERP_SHARE) and (interp_peaks < MAX_INTERP_SHARE)

                if keep:
            
                    pupil_pktr.append(pd.DataFrame({
                        "pupil_troughs":pupil_troughs,
                        "pupil_peaks":pupil_peaks,
                        "sub":datHash["sub"][i],
                        "Session":datHash["session"][i],
                        "Run":datHash["run"][i],
                        "Time":np.arange(pLen*2*cfg["RESAMPLING_RATE"])*(1/cfg["RESAMPLING_RATE"])-pLen
                        }))

            if keep:
        
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


    sTime = 0
    eTime = 7
    jitter=1

    event = "_troughs"

    df_tmp_all = df_tmp.copy()
    df_tmp = df_tmp_all.copy()
    df_tmp["wave_onset"] = df_tmp[f"wave_onset{event}_st"]
    df["wave_onset"] = df[f"wave_onset{event}_st"]


    df_tmp = df_tmp[(df_tmp["wave_onset"]<=eTime)]

    df_tmp_tmp = df_tmp.copy()
    df_tmp_tmp["P(T2|T1)_smooth_norm"] = -1

    df_ave_troughs=[]
    df_tmp_tmp["time_bin"]=-1

    for iSub in df_tmp_tmp["sub"].unique():
        for iSession in df_tmp_tmp["session"].unique():
            for iLag in df_tmp_tmp["lag"].unique():

                ind = (df_tmp_tmp["sub"]==iSub)&(df_tmp_tmp["session"]==iSession)&(df_tmp_tmp["lag"]==iLag)

                t = df_tmp_tmp[ind]["y_resid"].values

                df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = t - t.mean()
            
                t = df_tmp_tmp[ind]["P(T2|T1)"].values
                df_tmp_tmp.loc[ind,"P(T2|T1)_smooth_norm"] = t - t.mean()
            
                t = df_tmp_tmp[ind]["RT_t2_resid"].values
                df_tmp_tmp.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t)
            
                for iTarget in df_tmp_tmp["condition"].unique():
                
                    ind = (df["sub"]==iSub)&(df["session"]==iSession)&(df["lag"]==iLag)&(df["wave_onset"]<=eTime)&(df["condition"]==iTarget)
                       
                    t = df[ind]["y_resid"].values
                    df.loc[ind,"y_resid_smooth_norm"] = t - t.mean()
                            
                    t = df[ind]["P(T2|T1)"].values
                    df.loc[ind,"P(T2|T1)_smooth_norm"] = t - t.mean()

                    t = df[ind]["RT_t2_resid"].values            
                    df.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t)
            

    for iTime in np.arange(sTime,eTime-jitter,jitter):

        t = df_tmp_tmp[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter)].copy()
        df_tmp_tmp.loc[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter),"time_bin"]=iTime
    
        df.loc[(df["wave_onset"]>iTime)&(df["wave_onset"]<iTime+jitter),"time_bin"] = iTime
    
        del t["condition"]
        t = t.groupby(["sub","session","lag"],sort=False,as_index=False).agg(np.nanmean)

        t["time_bin"] = iTime
        df_ave_troughs.append(t)
    
    df_ave_troughs = pd.concat(df_ave_troughs).reset_index(drop=True)
    # trial-level values (one row per trial) for the per-bin LMM in figure.Rmd
    trials_troughs = df_tmp_tmp.reset_index(drop=True)


    sTime = 0
    eTime = 7
    jitter=1

    event = "_peaks"

    df_tmp = df_tmp_all.copy()
    df_tmp["wave_onset"] = df_tmp[f"wave_onset{event}_st"]
    df["wave_onset"] = df[f"wave_onset{event}_st"]


    df_tmp = df_tmp[(df_tmp["wave_onset"]<=eTime)]

    df_tmp_tmp = df_tmp.copy()
    df_tmp_tmp["P(T2|T1)_smooth_norm"] = -1

    df_ave_peaks=[]
    df_tmp_tmp["time_bin"]=-1

    for iSub in df_tmp_tmp["sub"].unique():
        for iSession in df_tmp_tmp["session"].unique():
            for iLag in df_tmp_tmp["lag"].unique():

                ind = (df_tmp_tmp["sub"]==iSub)&(df_tmp_tmp["session"]==iSession)&(df_tmp_tmp["lag"]==iLag)

                t = df_tmp_tmp[ind]["y_resid"].values

                df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = t - t.mean()
            
                t = df_tmp_tmp[ind]["P(T2|T1)"].values
                df_tmp_tmp.loc[ind,"P(T2|T1)_smooth_norm"] = t - t.mean()
            
                t = df_tmp_tmp[ind]["RT_t2_resid"].values
                df_tmp_tmp.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t)
            
                for iTarget in df_tmp_tmp["condition"].unique():
                
                    ind = (df["sub"]==iSub)&(df["session"]==iSession)&(df["lag"]==iLag)&(df["wave_onset"]<=eTime)&(df["condition"]==iTarget)
                       
                    t = df[ind]["y_resid"].values
                    df.loc[ind,"y_resid_smooth_norm"] = t - t.mean()
                            
                    t = df[ind]["P(T2|T1)"].values
                    df.loc[ind,"P(T2|T1)_smooth_norm"] = t - t.mean()

                    t = df[ind]["RT_t2_resid"].values            
                    df.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t)
            

    for iTime in np.arange(sTime,eTime-jitter,jitter):

        t = df_tmp_tmp[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter)].copy()
        df_tmp_tmp.loc[(df_tmp_tmp["wave_onset"]>iTime)&(df_tmp_tmp["wave_onset"]<iTime+jitter),"time_bin"]=iTime
    
        df.loc[(df["wave_onset"]>iTime)&(df["wave_onset"]<iTime+jitter),"time_bin"] = iTime
    
        del t["condition"]
        t = t.groupby(["sub","session","lag"],sort=False,as_index=False).agg(np.nanmean)

        t["time_bin"] = iTime
        df_ave_peaks.append(t)
    
    df_ave_peaks = pd.concat(df_ave_peaks).reset_index(drop=True)
    # trial-level values (one row per trial) for the per-bin LMM in figure.Rmd
    trials_peaks = df_tmp_tmp.reset_index(drop=True)
    return df, df_ave_troughs, df_ave_peaks, trials_troughs, trials_peaks, pupil_pktr


df_before_phase = df
df, df_ave_troughs, df_ave_peaks, trials_troughs, trials_peaks, pupil_pktr = phase_analysis(df_before_phase, PCPDevents)
trial_cols = ["sub","session","lag","condition","wave_onset","time_bin","y_resid_smooth_norm"]
df_trial_troughs = trials_troughs[trial_cols]
df_trial_peaks = trials_peaks[trial_cols]


# %% classify based on pupil

mmName="EventPD_t1"

def groupwise_qcut(x):
    try:
        return pd.qcut(x, q=5, labels=False, duplicates='drop')
    except ValueError:
        return pd.Series([np.nan] * len(x), index=x.index)

df['PD_bin'] = (
    df.groupby(['sub', 'session', 'lag','condition'])[mmName]
    .transform(groupwise_qcut)
)


# %% P(T2|T1) based on pupil size


pupil_pktr = pupil_pktr.groupby(["sub","Time","Session"], sort=False, as_index=False).agg(np.nanmean)


df.to_json(f"{savefolder}/df.json")
df_ave_troughs.to_json(f"{savefolder}/df_bin_troughs.json")
df_ave_peaks.to_json(f"{savefolder}/df_bin_peaks.json")
df_trial_troughs.to_json(f"{savefolder}/df_trial_troughs.json")
df_trial_peaks.to_json(f"{savefolder}/df_trial_peaks.json")
pupil_pktr.to_parquet(f"{savefolder}/df_pupilTimeCourse.parquet")

df_pupil = df_pupil[(df_pupil["condition"]=="Both")]
df_pupil.to_json(f"{savefolder}/df_pupil.json")




#%% sensitivity of the phase effect to the low-pass cutoff (Supplementary Figure 3)

CUTOFFS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 1, 2, 5]

sweep = []
for cut in CUTOFFS:
    print(f"cutoff sweep: {cut} Hz")
    if cut == bpass:
        ev_cut, bins_cut = PCPDevents, {"troughs": df_ave_troughs, "peaks": df_ave_peaks}
    else:
        _, ev_cut = detect_events(cut)
        _, bins_troughs, bins_peaks, _, _, _ = phase_analysis(df_before_phase, ev_cut)
        bins_cut = {"troughs": bins_troughs, "peaks": bins_peaks}

    rows = []
    for ev in ["troughs", "peaks"]:
        d = bins_cut[ev]
        d = d[d["time_bin"] >= 0].groupby(["sub", "session", "time_bin"], sort=False, as_index=False).mean(numeric_only=True)
        for ses in d["session"].unique():
            for tb in sorted(d["time_bin"].unique()):
                v = d[(d["session"] == ses) & (d["time_bin"] == tb)]["y_resid_smooth_norm"].to_numpy(float)
                v = v[np.isfinite(v)]
                t, p = ttest_1samp(v, 0)
                rows.append(dict(cutoff_hz=cut, event=ev, session=ses, time_bin=int(tb), n_sub=len(v),
                                 mean=v.mean(), sem=v.std(ddof=1)/np.sqrt(len(v)), t=t, p=p))
    out = pd.DataFrame(rows)
    # number of events and median inter-event interval (s) within runs
    for ev in ["peaks", "troughs"]:
        ints = [np.diff(np.sort(np.array(k)))/cfg["RESAMPLING_RATE"] for k in ev_cut[ev] if len(k) > 1]
        out[f"n_{ev}_total"] = int(sum(len(k) for k in ev_cut[ev]))
        out[f"median_interval_{ev}_s"] = np.median(np.concatenate(ints))
    sweep.append(out)

sweep = pd.concat(sweep).sort_values(["session", "event", "cutoff_hz", "time_bin"]).reset_index(drop=True)
# FDR (Benjamini-Hochberg) across the time bins of each cutoff x event x session
sweep["p_fdr"] = sweep.groupby(["cutoff_hz", "event", "session"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
sweep.to_csv(f"{savefolder}/cutoff_sweep_allbins.csv", index=False)


#%% T2-evoked pupil response and logistic models of T2 detection (Supplementary Material)

BASELINE_WIN = (-0.2, 0.0)
EVOKED_WIN = (0.5, 2.0)

def zscore_by_session(d, col):
    return d.groupby("session")[col].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

def trial_evoked_table():
    """Mean T2-evoked pupil change (EVOKED_WIN, relative to BASELINE_WIN) of each T2_only / Both trial"""
    c = {"TIME_START": -3, "TIME_END": 3, "RESAMPLING_RATE": 200} | json.load(open(f"{folderName}/cfg.json"))
    fs = int(c["RESAMPLING_RATE"])
    t = np.arange(int((c["TIME_END"] - c["TIME_START"]) * fs)) / fs + c["TIME_START"]
    baseline_mask = (t >= BASELINE_WIN[0]) & (t <= BASELINE_WIN[1])
    evoked_mask = (t >= EVOKED_WIN[0]) & (t <= EVOKED_WIN[1])
    cond_name = {3: "T2_only", 4: "Both"}

    rows = []
    for path in sorted(glob.glob(f"{folderName}/*_trial.msgpack")):
        with open(path, "rb") as f:
            d = msgpack.unpackb(f.read(), raw=False)
        for i in range(len(d["sub"])):
            target = int(d["target"][i])
            if target not in (3, 4):
                continue
            yy = np.asarray(d["PDR_t2"][i], dtype=float)
            if yy.size != t.size or not np.isfinite(yy).any():
                continue
            y_bc = yy - float(np.nanmean(yy[baseline_mask]))
            hit_t2 = int(np.asarray(d["responses"][i])[1] == 1)
            rows.append({
                "sub": str(d["sub"][i]), "session": str(d["session"][i]), "run": int(d["run"][i]),
                "condition": cond_name[target], "target": target, "lag": int(d["lag"][i]), "time": float(d["time"][i]),
                "hit_t2": hit_t2, "t2_outcome": "detected" if hit_t2 else "missed",
                "t2_evoked_auc_0p5_2s": float(np.nanmean(y_bc[evoked_mask])),
                "t2_evoked_peak_0p5_2s": float(np.nanmax(y_bc[evoked_mask])),
            })
    return pd.DataFrame(rows)

def fit_logit_models(data):
    """Logistic models of T2 detection per session and likelihood-ratio tests between them"""
    model_defs = [
        ("lag_subject", "hit_t2 ~ C(lag) + C(sub)"),
        ("pre_target", "hit_t2 ~ C(lag) + C(sub) + baseline_pd_z + C(phase_bin)"),
        ("baseline_plus_post_evoked", "hit_t2 ~ C(lag) + C(sub) + baseline_pd_z + t2_evoked_auc_z"),
        ("pre_plus_post_evoked", "hit_t2 ~ C(lag) + C(sub) + baseline_pd_z + C(phase_bin) + t2_evoked_auc_z"),
    ]
    data = data.copy()
    for col in ["hit_t2", "lag", "run", "target", "time_bin"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["sub", "phase_bin", "condition", "session"]:
        data[col] = pd.Series(list(map(str, data[col])), index=data.index, dtype=object)

    model_rows, coef_rows, fitted = [], [], {}
    for session, gd in data.groupby("session"):
        gd = gd.dropna(subset=["hit_t2", "baseline_pd_z", "t2_evoked_auc_z"])
        for name, formula in model_defs:
            r = smf.logit(formula, data=gd).fit(disp=False, maxiter=200)
            fitted[(session, name)] = r
            model_rows.append({"session": session, "session_label": session, "model": name, "n_trials": int(r.nobs),
                               "df_model": float(r.df_model), "log_likelihood": float(r.llf), "aic": float(r.aic),
                               "bic": float(r.bic), "pseudo_r2_mcfadden": float(r.prsquared)})
            for term, estimate in r.params.items():
                if term not in {"baseline_pd_z", "t2_evoked_auc_z"} and not term.startswith("C(phase_bin)"):
                    continue
                coef_rows.append({"session": session, "session_label": session, "model": name, "term": term,
                                  "estimate": float(estimate), "se": float(r.bse[term]), "z": float(r.tvalues[term]),
                                  "p": float(r.pvalues[term])})

    comparison_rows = []
    for session in data["session"].unique():
        for reduced, full, label in [
            ("lag_subject", "pre_target", "add_pre_target_PD_phase"),
            ("pre_target", "pre_plus_post_evoked", "add_post_T2_evoked_auc"),
            ("baseline_plus_post_evoked", "pre_plus_post_evoked", "phase_after_post_evoked_auc"),
        ]:
            r, f = fitted[(session, reduced)], fitted[(session, full)]
            lr = 2 * (f.llf - r.llf)
            df_diff = f.df_model - r.df_model
            comparison_rows.append({"session": session, "session_label": session, "comparison": label,
                                    "reduced": reduced, "full": full, "lr_chi2": float(lr), "df": float(df_diff),
                                    "p": float(chi2.sf(lr, df_diff)),
                                    "delta_aic_full_minus_reduced": float(f.aic - r.aic),
                                    "delta_bic_full_minus_reduced": float(f.bic - r.bic)})
    return pd.DataFrame(model_rows), pd.DataFrame(comparison_rows), pd.DataFrame(coef_rows)

evoked = trial_evoked_table()
keys = ["sub", "session", "run", "condition", "target", "lag", "time"]

## trials of the logistic models: phase bins 0-5 s of the trough-locked analysis (df.json)
d_model = pd.read_json(f"{savefolder}/df.json")
d_model = d_model[d_model["condition"].isin(["T2_only", "Both"])].copy()
d_model["target"] = d_model["condition"].map({"T2_only": 3, "Both": 4}).astype(int)
for col in ["run", "target", "lag"]:
    d_model[col] = d_model[col].astype(int)
d_model["time"] = d_model["time"].astype(float)
d_model = d_model.merge(evoked, on=keys + ["hit_t2"], how="inner", validate="one_to_one")
d_model = d_model[(d_model["time_bin"] >= 0) & (d_model["time_bin"] < 6)].copy()
d_model["phase_bin"] = d_model["time_bin"].astype(int).astype(str)
d_model["baseline_pd_z"] = zscore_by_session(d_model, "PD")
d_model["t2_evoked_auc_z"] = zscore_by_session(d_model, "t2_evoked_auc_0p5_2s")

model_table, comparison_table, coef_table = fit_logit_models(d_model)
model_table.to_csv(f"{savefolder}/stimulus_evoked_model_fit.csv", index=False)
comparison_table.to_csv(f"{savefolder}/stimulus_evoked_model_comparison.csv", index=False)
coef_table.to_csv(f"{savefolder}/stimulus_evoked_model_coefficients.csv", index=False)

## trial-level table for the Fig. 3B time-bin analysis with the T2-evoked response as covariate (figure.Rmd)
fig3b = trials_troughs.copy()
fig3b["target"] = fig3b["condition"].map({"T2_only": 3, "Both": 4}).astype(int)
fig3b = fig3b.merge(evoked[keys + ["t2_evoked_auc_0p5_2s"]], on=keys, how="left", validate="one_to_one")
fig3b[trial_cols + ["t2_evoked_auc_0p5_2s"]].to_csv(f"{savefolder}/stimulus_evoked_fig3b_trials.csv", index=False)
