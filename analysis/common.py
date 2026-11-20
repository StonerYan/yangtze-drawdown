"""Shared statistical and data-loading helpers for the Yangtze reservoir drawdown-zone analysis."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2 = os.path.join(ROOT, "data", "cee2")
DC = os.path.join(ROOT, "data", "cee")
DD = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
FIRST = ["Yalong_Lianghekou", "Lower_Jinsha_Baihetan", "Lower_Jinsha_Wudongde"]   # first impoundment inside the 2018-2025 record (commissioned 2020-2023)
YEARS = np.arange(2018, 2026)

def dump(obj, name):
    json.dump(obj, open(f"{OUT}/{name}.json", "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)

def ts_slope_rows(M, x):
    """Theil-Sen slope of each row of M (n x T) against x (T), ignoring NaN; needs >= 6 finite values."""
    n, T = M.shape; out = np.full(n, np.nan)
    ii, jj = np.triu_indices(T, 1); dx = x[jj] - x[ii]; ok = dx != 0
    ii, jj, dx = ii[ok], jj[ok], dx[ok]
    S = (M[:, jj] - M[:, ii]) / dx
    good = np.isfinite(M).sum(1) >= 6
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore")
        med = np.nanmedian(S, axis=1)
    out[good] = med[good]; return out

def boot_mean(v, B=5000, seed=1):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); v = v[np.isfinite(v)]
    b = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(len(v)), median=float(np.median(v)),
                med_lo=float(np.percentile(np.median(v[rng.integers(0, len(v), (B, len(v)))], 1), 2.5)),
                med_hi=float(np.percentile(np.median(v[rng.integers(0, len(v), (B, len(v)))], 1), 97.5)))

def block_boot_mean(v, blocks, B=5000, seed=2):
    """Cluster bootstrap of the mean over spatial blocks (resample blocks, keep reservoirs inside)."""
    rng = np.random.default_rng(seed); v = np.asarray(v, float); blocks = np.asarray(blocks)
    ub = np.unique(blocks); idx = {b: np.where(blocks == b)[0] for b in ub}; out = np.empty(B)
    for k in range(B):
        pick = rng.choice(ub, len(ub)); i = np.concatenate([idx[b] for b in pick]); out[k] = np.nanmean(v[i])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
