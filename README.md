# PupilPhase-AttentionalBlink

Analysis code and processed data for
*"Real-time tracking of pupil-phase fluctuations reveals state-dependent modulation of temporal attentional capacity"*
(Suzuki, Y. and Liao, H.-I.).

Copyright 2026 Yuta Suzuki. See `LICENSE`.

## Folder structure

```
lib/                 Python pre-processing modules (blink/artifact handling, au-to-mm conversion, filters, peak/trough detection)
                     and common_profile.R (R packages and helper functions)
                     and anovakun_487.R (ANOVA function used by common_profile.R)
                     and doc_templete.docx (Word template for the knitted documents)
Exp1/                Experiment 1 (offline, T1-attend / T1-ignore)
  parseData.py         raw .asc -> pre-processed pupil data (msgpack)
  dataAnalysis.py      epoching, artifact rejection, pupil-phase analysis -> data/<date>/*.json
                       also the low-pass cutoff sensitivity (cutoff_sweep_allbins.csv) and
                       the T2-evoked pupil response models (stimulus_evoked_*.csv)
  data/<date>/         processed data (the scripts and Rmd files read the latest <date> folder)
  subInfo.xlsx         participant age and gender
Exp2/                Experiment 2 (real-time pupil-phase triggering, PD / PC phase)
  parseData.py
  dataAnalysis.py      also writes the example traces for Supplementary Figure 4A
  data/<date>/
  subInfo.xlsx
figure.Rmd           all figures and statistics, writes figure/<date>/output/results_objects.rds
results.Rmd          statistics reported in the Results section (reads results_objects.rds)
supplementary.Rmd    Supplementary Material (reads results_objects.rds)
```

## Requirements

Python 3.12 with numpy, scipy, pandas, msgpack, statsmodels, joblib, scikit-learn and pyarrow.

R 4.5 with the packages listed in `loaded_packages_versions.csv`. To install the missing ones:

```r
pkg_list <- read.csv("loaded_packages_versions.csv", stringsAsFactors = FALSE)
install.packages(setdiff(pkg_list$Package, rownames(installed.packages())))
```

## Reproducing the figures and statistics

The processed data in `Exp1/data` and `Exp2/data` are sufficient. Knit `figure.Rmd` first,
then `results.Rmd` and `supplementary.Rmd`.

Open this folder as the workspace (e.g. in Positron). The R Markdown files use the repository root as the working
directory, and the Python scripts switch to their own folder, so chunks and cells can also be run interactively.

## Pre-processing from raw data

Raw data (EyeLink .edf/.asc, behavioral logs) are in `Exp1/rawData/T1-attend/sXX/`, `Exp1/rawData/T1-ignore/sXX/`
and `Exp2/rawData/sXX/`. The full pipeline is run with

```
python Exp1/parseData.py
python Exp1/dataAnalysis.py
python Exp2/parseData.py
python Exp2/dataAnalysis.py
```

`parseData.py` writes the pre-processed data to `data/<date>/` and `dataAnalysis.py` adds the files read by the R Markdown files to the latest folder in `data/`.
