#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
## run from this folder, also when executed from the workspace root (e.g. Positron)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath("Exp2" if os.path.isdir("Exp2") else ".")
os.chdir(HERE)
sys.path.append(os.path.join(HERE, "..", "lib"))

import glob
import json
import warnings

import msgpack
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from band_pass_filter import lowpass_filter
from joblib import Parallel, delayed
from pre_processing_cls import (
    getNearestValue,
    getPCPDevents,
    pre_processing,
    rejectDat,
)
from quickpsy import fit_quickpsy_group
from scipy.stats import zscore


warnings.simplefilter("ignore")


#%% ------------------ data loading -----------------------------------------

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
    

cfg["THRES_DIFF"] = 0.1
y_corrected = np.array(datHash["PDR_trial"])
y_corrected = y_corrected.reshape(-1,y_corrected.shape[2])


reject = {}
pp = pre_processing(cfg)

y,reject["PDR"] = pp.pre_processing(y_corrected)

rejectArray=np.zeros(y_corrected.shape[0])
rejectArray[reject["PDR"]]=1


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


#%% reject sub

def remove_outliers_mad(x, k=3.5, return_z=False):
    """Remove outliers by the modified z-score (MAD)."""
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

tmp_df = df
tmp_df = tmp_df.groupby(["sub","condition"],sort=False,as_index=False).agg("mean",numeric_only=True)

rejectSub=[]
for (c,h) in zip(["T1_only","T2_only","T2_only","T1_only"],["hit_t1","hit_t2","fa_t1","fa_t2"]):

    tmp_tmp_df = tmp_df[tmp_df["condition"]==c]

    
    mask = remove_outliers_mad(tmp_tmp_df[h], k=3.5, return_z=True)
    
    rejectSub.append(tmp_tmp_df[mask]["sub"].values)


rejectSub = np.concatenate(rejectSub)
rejectSub = sorted(set(rejectSub))

print(f"{rejectSub} was rejected because of low accuracy for T1 or T2")

for iSub in rejectSub:
    df = df[df["sub"]!=iSub]
    
reject = [i for i,d in enumerate(datHash["sub"]) if d in rejectSub]

datHash = rejectDat(datHash,reject)

#%% fraction of interpolated samples (zeroArray) in the included participants
zero_frac = pd.DataFrame({
    "sub": datHash["sub"],
    "zero_frac": [np.mean(np.asarray(z).reshape(-1)[np.isfinite(np.asarray(p, dtype=float).reshape(-1))])
                  for z, p in zip(datHash["zeroArray"], datHash["PDR"])],
}).groupby("sub")["zero_frac"].mean()
# summary for the "Pupil data quality" paragraph of results.Rmd
pd.DataFrame({"n_sub": [len(zero_frac)], "interp_frac_mean": [zero_frac.mean()],
              "interp_frac_std": [zero_frac.std()]}).to_csv(f"{folderName}/zero_frac_summary.csv", index=False)

#%% interval between successive triggered sound sequences
df_interval = []
for iRun in np.arange(len(datHash["PDR"])):
    onsets = np.sort(np.r_[np.array(datHash["t1"][iRun], dtype=float),
                           np.array(datHash["t2"][iRun], dtype=float)]) / cfg["SAMPLING_RATE"]
    seq_onsets = onsets[np.r_[True, np.diff(onsets) > 1.5]]
    df_interval.append(pd.DataFrame({
        "sub": datHash["sub"][iRun],
        "analysis": datHash["analysis"][iRun],
        "run": datHash["run"][iRun],
        "interval": np.diff(seq_onsets),
    }))
df_interval = pd.concat(df_interval).reset_index(drop=True)
df_interval.to_json(f"{folderName}/df_interval.json")


print("pre-processing...")

y_corrected = datHash["PDR"]

y_lowpassed = lowpass_filter(y_corrected, 0.15, cfg["RESAMPLING_RATE"])

y = [np.array(y).reshape(1,-1) for y in y_lowpassed]
y_mm = [np.array(y).reshape(1,-1) for y in y_corrected]
PCPDevents = getPCPDevents(y, 0.02)


pLen = 1.5
xx = np.arange(2*pLen*cfg["RESAMPLING_RATE"])*(1/cfg["RESAMPLING_RATE"])-pLen

df_pupil=[]
trialCount=0
for i,condition in enumerate(datHash["t1"]):

    pupil = zscore(y_mm[i].reshape(-1))
    pupil_mm = y_mm[i].reshape(-1)

    for idx in condition[:-1]:

        tmp_idx = int(idx*(cfg["RESAMPLING_RATE"]/cfg["SAMPLING_RATE"]))

        ev_ind = np.arange(tmp_idx-int(pLen*cfg["RESAMPLING_RATE"]),int(tmp_idx+pLen*cfg["RESAMPLING_RATE"]))
        if len(np.argwhere(abs(np.diff(pupil_mm[ev_ind]))>0.1))==0:

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

df_pupil = pd.concat(df_pupil).reset_index(drop=True)


mmName = [
    "amp_peaks_st",
    "amp_troughs_st",
    "wave_onset_peaks_st",
    "wave_onset_troughs_st",
    ]

wave_summary = []

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
        
        while (pcpd[idx["peak"]["back"]]<=0) & (idx["peak"]["back"]>0):
            idx["peak"]["back"] -= 1

        idx["trough"]["back"] = onset_idx
        
        while (pcpd[idx["trough"]["back"]]>=0) & (idx["trough"]["back"]>0):
            idx["trough"]["back"] -= 1
        
        for event_type in ["trough","peak"]:
            idx[f"amp_{event_type}_st"].append(np.round(pupil[idx[event_type]["back"]] - pupil[onset_idx], 4))
            idx[f"wave_{event_type}_st"].append(np.round(time[onset_idx] - time[idx[event_type]["back"]], 4))

    wave_summary.append(pd.DataFrame({
        "amp_peaks_st":idx["amp_peak_st"],
        "amp_troughs_st":idx["amp_trough_st"],
        "wave_onset_peaks_st":idx["wave_peak_st"],
        "wave_onset_troughs_st":idx["wave_trough_st"],
        "sub":datHash["sub"][i],
        "run":datHash["run"][i],
        "session":datHash["analysis"][i],
        }))
    
wave_summary = pd.concat(wave_summary).reset_index(drop=True)


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


event = "_troughs"
df_tmp=df.copy()
df_tmp["wave_onset"] = df_tmp[f"wave_onset{event}_st"]
df["wave_onset"] = df[f"wave_onset{event}_st"]


df_tmp = df_tmp[(df_tmp["wave_onset"]<=eTime)]

df_tmp_tmp = df_tmp.copy()
df_tmp_tmp["P(T2|T1)_smooth_norm"] = np.nan

for iSub in df_tmp_tmp["sub"].unique():
    for iSession in df_tmp_tmp["analysis"].unique():
        for iLag in df_tmp_tmp["lag_cond"].unique():

            ind = (df_tmp_tmp["sub"]==iSub)&(df_tmp_tmp["analysis"]==iSession)&(df_tmp_tmp["lag_cond"]==iLag)
            
            t = df_tmp_tmp[ind]["y_resid"].values
    
            df_tmp_tmp.loc[ind,"y_resid_smooth_norm"] = t - np.nanmean(t[t!=-100])
            
            t = df_tmp_tmp[ind]["P(T2|T1)"].values
            df_tmp_tmp.loc[ind,"P(T2|T1)_smooth_norm"] = t - np.nanmean(t[t!=-100])
            
            t = df_tmp_tmp[ind]["RT_t2_resid"].values
            df_tmp_tmp.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t[t!=-100])
            
            for iTarget in df_tmp_tmp["condition"].unique():
                
                ind = (df["sub"]==iSub)&(df["analysis"]==iSession)&(df["wave_onset"]<=eTime)&(df["condition"]==iTarget)&(df["lag_cond"]==iLag)
                       
                t = df[ind]["y_resid"].values
                df.loc[ind,"y_resid_smooth_norm"] = t - np.nanmean(t[t!=-100])
                            
                t = df[ind]["P(T2|T1)"].values
                df.loc[ind,"P(T2|T1)_smooth_norm"] = t - np.nanmean(t[t!=-100])
    
                t = df[ind]["RT_t2_resid"].values            
                df.loc[ind,"RT_t2_resid_norm"] = t - np.nanmean(t[t!=-100])
        

for iTime in np.arange(sTime,eTime-jitter,jitter):
    df.loc[(df["wave_onset"]>iTime)&(df["wave_onset"]<iTime+jitter),"time_bin"] = int(iTime)

# %% exclude rejected trials

df = df[df["reject"]==0]


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

group_cols = ["analysis"]
groups = list(df.groupby(group_cols))

def run_one_group(key, g):
    analysis = key
    
    out = fit_quickpsy_group(
        g,
        x_col="lag_cond",
        y01_col="P(T2|T1)",
        grouping_values={"analysis": analysis},
        prob=0.5,
        B=10000,
        seed=42
    )
    return out

outs = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
    delayed(run_one_group)(key, g.copy()) for key, g in groups
)

outs = [o for o in outs if o is not None]


boot_thr = []
for o in outs:
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


df.to_json(f"{folderName}/df.json")

boot_thr_df.to_json(f"{folderName}/df_pse_thre.json")

df_pupil_ave = df_pupil.groupby(["sub","analysis","time"],sort=False,as_index=False).agg("mean",numeric_only=True)

df_pupil_ave.to_json(f"{folderName}/df_pupil.json")


#%% example pupil traces for Supplementary Figure 4A (drawn in figure.Rmd)

def read_messages(asc_file, keys):
    out = {k: [] for k in keys}
    with open(asc_file, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("MSG"):
                continue
            parts = line.split()
            for k in keys:
                if len(parts) >= 3 and parts[2] == k:
                    out[k].append(int(parts[1]))
    return out


def export_example_trace(folderName, sub="s24", runs={"PD": 4, "PC": 2}, cutoff=0.15, prominence=0.02):
    """runs: run index in <sub>_run.msgpack (SBP = PD, LBP = PC)"""
    fs_raw = cfg["SAMPLING_RATE"]
    fs = cfg["RESAMPLING_RATE"]
    with open(f"{folderName}/{sub}_run.msgpack", "rb") as f:
        dat = msgpack.unpackb(f.read(), raw=False)
    # parseData.py numbers the runs in sorted(glob(f"{iSub}/**/*.asc")) order
    asc_files = sorted(glob.glob(f"./rawData/{sub}/**/*.asc"))
    assert len(asc_files) == len(dat["PDR"]), "run count mismatch between asc files and msgpack"

    trace, events = [], []
    for phase, iRun in runs.items():
        mode = dat["analysis"][iRun]
        assert (phase == "PD" and mode == "SBP") or (phase == "PC" and mode == "LBP"), (phase, mode)

        pupil = np.array(dat["PDR"][iRun], dtype=float)
        finite = np.isfinite(pupil)
        filled = pupil.copy()
        filled[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), pupil[finite])
        pupil_lp = np.array(lowpass_filter(filled, cutoff, fs))
        time = np.arange(len(pupil)) / fs

        ev = getPCPDevents([pupil_lp.reshape(1, -1)], prominence)
        extrema = np.array(ev["troughs" if phase == "PD" else "peaks"][0], dtype=int)

        msg = read_messages(asc_files[iRun], ["ONSET_T1", "TROUGH_DETECTED"])
        t1_rel = np.array(dat["t1"][iRun], dtype=float)
        t1_abs = np.array(msg["ONSET_T1"], dtype=float)
        assert len(t1_abs) == len(t1_rel), "T1 count mismatch"
        offset = np.median(t1_abs - t1_rel)
        assert np.all(np.abs((t1_abs - t1_rel) - offset) < 2), "inconsistent message offset"
        trig = (np.array(msg["TROUGH_DETECTED"], dtype=float) - offset) / fs_raw
        t1 = t1_rel / fs_raw

        trace.append(pd.DataFrame({"sub": sub, "phase": phase, "run": iRun,
                                   "time": time, "pupil": pupil, "pupil_lp": pupil_lp}))
        events.append(pd.concat([
            pd.DataFrame({"event": "extremum", "time": time[extrema], "pupil_lp": pupil_lp[extrema]}),
            pd.DataFrame({"event": "trigger", "time": trig, "pupil_lp": np.interp(trig, time, pupil_lp)}),
            pd.DataFrame({"event": "T1", "time": t1, "pupil_lp": np.interp(t1, time, pupil_lp)}),
        ]).assign(sub=sub, phase=phase, run=iRun))

    pd.concat(trace).reset_index(drop=True).to_json(f"{folderName}/df_example_trace_exp2.json")
    pd.concat(events).reset_index(drop=True).to_json(f"{folderName}/df_example_events_exp2.json")


export_example_trace(folderName)
