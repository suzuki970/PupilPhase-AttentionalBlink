## Experimental data for *"Real-time tracking of pupil-phase fluctuations reveals state-dependent modulation of temporal attentional capacity"*
Copyright 2026 Yuta Suzuki


### Article information
Suzuki, Y and Liao, H.

## Requirements
Python
- pre-peocessing (**'/[Python]PreProcessing/PupilAnalysisToolbox'**)
- numpy
- scipy
- os
- json

R

Run the following code first to install the required packages.
```
pkg_list <- read.csv("loaded_packages_versions.csv", stringsAsFactors = FALSE)

for (i in 1:nrow(pkg_list)) {
  if (!pkg_list[i,"Package"] %in% installed.packages()[,"Package"]){
    install.packages(pkg_list[i,"Package"])
  }
}
```
## Raw data
Raw data can be found at **Exp1/rawData'** and **Exp2/rawData'**

## Pre-processing
- Raw data (.asc) are pre-processed by **'ExpX/parseData.py'**

	- Pre- processed data is saved as **‘s[subject ID]_trial.json’**
	
- Artifact rejection and data epoch are performed by **'ExpX/dataAnalysis.py'**

## Figure and statistics
- *‘[Rmd]Results/figure_all2.Rmd’* is to generate figures and statistical results.


### Article information
  
