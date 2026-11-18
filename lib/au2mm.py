import glob
import os
from functools import lru_cache

import numpy as np

## calibration recordings of artificial pupils (printed circles) with the EyeLink
CALIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "au2mm", "Eyelink", "area", "700")
LR_CALIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "au2mm", "Eyelink", "diameter", "600")


def calib_dist(calib_dir):
    """viewing distance (mm) of a calibration folder, from its name (e.g. .../700 -> 700)"""
    return int(os.path.basename(os.path.normpath(calib_dir)))


def _read_calib(file):
    """mean pupil value of samples [5000:10000], PUPIL mode and recorded eye of one .asc"""
    tmp, mode, eye = [], None, None
    with open(file) as f:
        for i, line in enumerate(f):
            line = line.split()
            if line[:1] == ["PUPIL"]:
                mode = line[1]
            if line[:1] == ["SAMPLES"]:
                eye = "L" if "LEFT" in line else "R"
            if i > 30 and line and line[0].isdecimal():
                tmp.append(float(line[3]))
    return float(np.mean(tmp[5000:10000])), mode, eye


@lru_cache(maxsize=None)
def lr_factor():
    """left-eye coefficient / right-eye coefficient, from the 7 mm circle in LR_CALIB_DIR"""
    d = {}
    for eye, name in (("L", "7mm_left.asc"), ("R", "7mm_right.asc")):
        value, mode, rec_eye = _read_calib(os.path.join(LR_CALIB_DIR, name))
        if rec_eye != eye or mode != "DIAMETER":
            raise ValueError(f"{name}: expected eye {eye} / DIAMETER, got {rec_eye} / {mode}")
        d[eye] = value
    return d["R"] / d["L"]


@lru_cache(maxsize=None)
def calib_coef():
    """mm per sqrt(area a.u.), from the calibration recordings in CALIB_DIR"""
    sizes, sqrtA = [], []
    for file in sorted(glob.glob(os.path.join(CALIB_DIR, "*.asc"))):
        value, mode, _ = _read_calib(file)
        if mode != "AREA":
            raise ValueError(f"{file}: expected PUPIL AREA, got {mode}")
        sizes.append(int(os.path.basename(file)[0]))
        sqrtA.append(np.sqrt(value))
    return float(np.mean(np.array(sizes, float) / np.array(sqrtA)))


def au2mm(pupilData, mode, eye, vis_dist):
    """Convert EyeLink pupil size (a.u.) of eye 'L' or 'R' to diameter in mm"""
    p = np.abs(np.asarray(pupilData, dtype=float))

    k = calib_coef() * (lr_factor() if eye == "L" else 1.0)
    k = k * vis_dist / calib_dist(CALIB_DIR)

    if mode == "AREA":
        return np.sqrt(p) * k
    if mode == "DIAMETER":
        return p * k * np.sqrt(np.pi) / 2
    raise ValueError(f"mode must be 'AREA' or 'DIAMETER', got {mode!r}")
