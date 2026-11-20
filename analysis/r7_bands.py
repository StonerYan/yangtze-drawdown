"""R7 - Drawdown-zone trend by surface-water occurrence band (median peak EVI2 of pixels in the band), 2018-2025, 71 established reservoirs and 3 first-filling reservoirs.
Also the window-level share of drawdown pixels in each band and the mean level of peak EVI2 per band."""
import glob, warnings; warnings.filterwarnings("ignore")
from common import *
pix = pd.read_parquet(f"{DC}/pixels.parquet"); pix = pix[pix.cls == "dz"]; Y = np.arange(2018, 2026); rows = []; lev = []
BANDS = [(15, 35), (35, 55), (55, 75), (75, 95.01)]
for w, g in pix.groupby("window"):
    pk = pd.read_parquet(f"{DC}/peaks/{w}.parquet"); pk = pk[pk.year.between(2018, 2025)].pivot(index="pid", columns="year", values="peak").reindex(g.pid.values)[list(Y)]; M = pk.values; F = g.F.values
    ring = pd.read_parquet(f"{DC}/peaks/{w}.parquet")
    for lo, hi in BANDS:
        m = (F >= lo) & (F < hi)
        if m.sum() < 20: continue
        s = np.array([np.nanmedian(M[m, j]) if np.isfinite(M[m, j]).sum() >= 20 else np.nan for j in range(len(Y))])
        rows.append(dict(window=w, reservoir=g.reservoir.iat[0], lo=lo, S=ts_slope_rows(s[None, :], Y.astype(float))[0] * 1e3, level=np.nanmean(s), npix=int(m.sum())))
B = pd.DataFrame(rows); B.to_csv(f"{OUT}/band_window_slopes.csv", index=False)
RS = pd.read_csv(f"{OUT}/res18.csv").set_index("reservoir"); out = {}
for lab, sel in (("n71", ~B.reservoir.isin(FIRST)), ("n3", B.reservoir.isin(FIRST))):
    out[lab] = {}
    for lo, g in B[sel].groupby("lo"):
        rm = g.groupby("reservoir")[["S", "level", "npix"]].mean(); b = boot_mean(rm.S.values, B=4000)
        r_ = RS.loc[rm.index, "R"]; b["R_same_reservoirs"] = float(r_.mean()); b["G"] = boot_mean((rm.S - r_).dropna().values, B=4000)
        b["level"] = float(rm.level.mean()); b["n_res"] = int(len(rm)); b["pixel_share"] = float(g.npix.sum() / B[sel].npix.sum()); out[lab][int(lo)] = b
dump(out, "r7_bands")
for lab, d in out.items():
    for lo, b in d.items(): print(lab, lo, "S %.2f (%.2f,%.2f) med %.2f | R %.2f | G %.2f (%.2f,%.2f) | level %.3f | n %d | share %.2f" % (b["mean"], b["lo"], b["hi"], b["median"], b["R_same_reservoirs"], b["G"]["mean"], b["G"]["lo"], b["G"]["hi"], b["level"], b["n_res"], b["pixel_share"]))
