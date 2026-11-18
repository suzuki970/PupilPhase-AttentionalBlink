import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def nll_binomial(params, x, k, n):
    """Negative log-likelihood of a cumulative-normal psychometric function (guess = lapse = 0)"""
    p1, log_p2 = params
    p2 = np.exp(log_p2) + 1e-12
    psi = np.clip(norm.cdf((x - p1) / p2), 1e-9, 1-1e-9)
    return -(k * np.log(psi) + (n - k) * np.log(1 - psi)).sum()


def fit(x, k, n):
    theta0 = np.array([np.median(x), np.log(np.std(x) if np.std(x) > 0 else 1.0)])
    res = minimize(nll_binomial, theta0, args=(x, k, n), method="Nelder-Mead")
    if not res.success:
        res = minimize(nll_binomial, theta0, args=(x, k, n), method="Powell")
    return res


def fit_quickpsy_group(df_group, x_col, y01_col, grouping_values=None, prob=0.75, B=1000, seed=0):
    """Fit a cumulative-normal psychometric function to trials with y in {0, 1} (as quickpsy in R)
    and return the threshold at `prob` and its parametric-bootstrap distribution (B resamples)"""
    rng = np.random.default_rng(seed)

    ## aggregate to x level: k successes, n trials
    d = df_group[[x_col, y01_col]].dropna().copy()
    d = d.assign(_y=d[y01_col].astype(int))
    agg = d.groupby(x_col, as_index=False).agg(
        k=("_y", "sum"),
        n=("_y", "size"),
    ).rename(columns={x_col: "x"})

    if len(agg) < 3 or agg["x"].nunique() < 2:
        return None

    x = agg["x"].astype(float).to_numpy()
    k = agg["k"].astype(float).to_numpy()
    n = agg["n"].astype(float).to_numpy()

    res = fit(x, k, n)
    if not res.success:
        return None
    p1_hat, logp2_hat = res.x
    p2_hat = float(np.exp(logp2_hat))

    ## threshold x* such that psi(x*) = prob, within [0, max lag]
    inv = norm.ppf(np.clip(prob, 1e-9, 1-1e-9))
    thr = float(np.clip(p1_hat + p2_hat * inv, 0, float(agg["x"].max())))

    thresholds = pd.DataFrame([{**(grouping_values or {}), "prob": prob, "threshold": thr}])

    ## parametric bootstrap: k ~ Binomial(n, psi(x))
    thrs = []
    for b in range(B):
        psi_hat = norm.cdf((x - p1_hat)/p2_hat)
        kb = rng.binomial(n.astype(int), np.clip(psi_hat, 1e-9, 1-1e-9))
        rb = fit(x.copy(), kb, n.copy())
        if not rb.success:
            continue
        p1b, logp2b = rb.x
        thrs.append(p1b + float(np.exp(logp2b)) * inv)

    return {"thresholds": thresholds, "boot_thresholds": np.asarray(thrs, dtype=float)}
