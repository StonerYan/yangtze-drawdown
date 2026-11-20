"""R4 - Pixel-level analyses: (i) comparability of drawdown zone and upland control (R1.1); (ii) sensitivity of S and R to the
occurrence thresholds, to vegetated-pixel and cropland screening (R1.3, R3, R2-7b); (iii) vegetated-fraction thresholds (R2-8);
(iv) valid-pixel fractions by year.  Window-year medians -> Theil-Sen (>=6 yr, 2018-2025) -> reservoir mean."""
import glob, warnings, numpy as np, pandas as pd
from common import *
warnings.filterwarnings("ignore")
pix = pd.read_parquet(f"{DC}/pixels.parquet"); att = pd.read_parquet(f"{DC}/pixel_attrs.parquet")
P = pix.merge(att, on="pid", how="left")
win = pd.read_csv(f"{DC}/windows.csv")
R0 = pd.read_parquet(f"{D2}/res_table.parquet")
# ---------- (i) comparability ----------
cmp_rows = []
P["band"] = np.where(P.cls == "dz", "dz", pd.cut(P.dist, [59, 300, 600, 1100], labels=["60-300", "300-600", "600-1050"]).astype(str))
dzmed = P[P.cls == "dz"].groupby("window").elev.median().rename("dz_elev")
P = P.join(dzmed, on="window"); P["rel_elev"] = P.elev - P.dz_elev
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}
for b, g in P[P.band != "nan"].groupby("band"):
    r = dict(band=b, n_pix=int(len(g)), n_win=int(g.window.nunique()), slope_med=float(g.slope.median()), slope_q25=float(g.slope.quantile(.25)), slope_q75=float(g.slope.quantile(.75)),
             rel_elev_med=float(g.rel_elev.median()), rel_elev_q25=float(g.rel_elev.quantile(.25)), rel_elev_q75=float(g.rel_elev.quantile(.75)))
    vc = g.wc2020.map(WC).fillna("other").value_counts(normalize=True)
    for k in ["tree", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "other"]: r["wc_" + k] = float(vc.get(k, 0))
    r["stable_cci_wc"] = float(((g.cci2000 == g.cci2010) & (g.cci2010 == g.cci2020) & (g.wc2020 == g.wc2021)).mean())
    cmp_rows.append(r)
CMP = pd.DataFrame(cmp_rows); CMP.to_csv(f"{OUT}/comparability_pixels.csv", index=False); print(CMP.round(3).to_string())
# window-level matching of slope distribution: median of window medians
wl = P[P.band != "nan"].groupby(["window", "band"]).agg(slope=("slope", "median"), rel=("rel_elev", "median"),
        crop=("wc2020", lambda s: (s == 40).mean()), built=("wc2020", lambda s: (s == 50).mean()), tree=("wc2020", lambda s: (s == 10).mean())).reset_index()
wlm = wl.groupby("band")[["slope", "rel", "crop", "built", "tree"]].median(); wlm.to_csv(f"{OUT}/comparability_window_medians.csv"); print(wlm.round(3))
# ---------- (ii)-(iv) sensitivity ----------
pk = pd.concat([pd.read_parquet(f).assign(window=os.path.basename(f)[:-8]) for f in sorted(glob.glob(f"{DC}/peaks/*.parquet"))])
Y = np.arange(2017, 2026); x18 = Y[1:]
VARS = {"base": None, "dz_25_95": ("dz", 25, 95), "dz_35_95": ("dz", 35, 95), "dz_15_75": ("dz", 15, 75), "dz_15_55": ("dz", 15, 55),
        "land_F_lt5": ("land", 0, 5), "ever_vegetated": "ever", "no_crop_built": "nocrop", "land_hi_slope_free": "slope<25"}
res_rows, vf_rows, valid_rows = [], [], []
for w, gp in P.groupby("window"):
    g = gp.set_index("pid"); pw = pk[pk.window == w].pivot(index="pid", columns="year", values="peak").reindex(g.index)[Y]
    nobs = pk[pk.window == w].pivot(index="pid", columns="year", values="nobs").reindex(g.index)[Y]
    M = pw.values; dz = (g.cls == "dz").values; ring = ((g.cls == "land") & g.dist.between(60, 300)).values
    F = g.F.values; wc = g.wc2020.values; sl = g.slope.values
    def med(mask):
        out = np.full(len(Y), np.nan)
        for j in range(len(Y)):
            v = M[mask, j]; v = v[np.isfinite(v)]
            if len(v) >= 20: out[j] = np.median(v)
        return out
    ever = np.nanmax(np.where(np.isfinite(M), M, -1), 1) >= 0.2
    notcb = ~np.isin(wc, [40, 50])
    sets = {"base": (dz, ring), "dz_25_95": (dz & (F >= 25), ring), "dz_35_95": (dz & (F >= 35), ring), "dz_15_75": (dz & (F <= 75), ring), "dz_15_55": (dz & (F <= 55), ring),
            "land_F_lt5": (dz, ring & (F < 5)), "ever_vegetated": (dz & ever, ring & ever), "no_crop_built": (dz & notcb, ring & notcb), "land_hi_slope_free": (dz, ring & (sl < 25))}
    for k, (md, mr) in sets.items():
        s, r = med(md), med(mr)
        res_rows.append(dict(window=w, reservoir=g.reservoir.iat[0], variant=k, S=ts_slope_rows(s[None, 1:], x18)[0], R=ts_slope_rows(r[None, 1:], x18)[0],
                             G=ts_slope_rows((s - r)[None, 1:], x18)[0], n_dz=int(md.sum()), n_ring=int(mr.sum())))
    # vegetated fraction by threshold and occurrence band
    bands = [(15, 35), (35, 55), (55, 75), (75, 95.01), (15, 95.01)]
    for thr in (0.15, 0.2, 0.3):
        for lo, hi in bands:
            m = dz & (F >= lo) & (F < hi)
            if m.sum() < 20: continue
            fr = np.array([np.nanmean(M[m, j] >= thr) if np.isfinite(M[m, j]).sum() >= 20 else np.nan for j in range(len(Y))])
            vf_rows.append(dict(window=w, reservoir=g.reservoir.iat[0], thr=thr, lo=lo, hi=hi, V_slope=ts_slope_rows(fr[None, 1:], x18)[0], V_mean=np.nanmean(fr[1:]), npix=int(m.sum())))
    # valid pixel fraction by year (dz and ring)
    for j, y in enumerate(Y):
        valid_rows.append(dict(window=w, reservoir=g.reservoir.iat[0], year=int(y), valid_dz=float(np.isfinite(M[dz, j]).mean()), valid_ring=float(np.isfinite(M[ring, j]).mean()),
                               nobs_dz=float(np.nanmean(nobs.values[dz, j])), nobs_ring=float(np.nanmean(nobs.values[ring, j]))))
RES = pd.DataFrame(res_rows); RES.to_parquet(f"{OUT}/sens_window_slopes.parquet")
VF = pd.DataFrame(vf_rows); VF.to_parquet(f"{OUT}/vegfrac_window_slopes.parquet")
VAL = pd.DataFrame(valid_rows); VAL.to_parquet(f"{OUT}/valid_fraction_winyear.parquet")
# ---- summaries: reservoir mean -> bootstrap, for all 74 and 71 ----
summ = {}
for k, g in RES.groupby("variant"):
    rm = g.groupby("reservoir")[["S", "R", "G"]].mean(); d = {}
    for sel, idx in (("n74", rm.index), ("n71", rm.index.difference(FIRST))):
        d[sel] = {v: boot_mean(rm.loc[idx, v].values * 1e3, B=3000) for v in ["S", "R", "G"]}
    summ[k] = d
dump(summ, "sensitivity_variants")
for k, d in summ.items():
    print(k, "n71 S %.2f R %.2f G %.2f | n74 S %.2f R %.2f G %.2f" % (d["n71"]["S"]["mean"], d["n71"]["R"]["mean"], d["n71"]["G"]["mean"], d["n74"]["S"]["mean"], d["n74"]["R"]["mean"], d["n74"]["G"]["mean"]))
vs = {}
for (thr, lo, hi), g in VF.groupby(["thr", "lo", "hi"]):
    rm = g.groupby("reservoir").V_slope.mean() * 100
    vs[f"{thr}|{lo}-{int(hi)}"] = dict(n74=boot_mean(rm.values, B=3000), n71=boot_mean(rm.drop(FIRST, errors="ignore").values, B=3000), Vmean=float(g.V_mean.mean()))
dump(vs, "vegfrac_thresholds")
for k, d in vs.items(): print(k, "n74 %.2f (%.2f,%.2f) n71 %.2f (%.2f,%.2f) mean V %.2f" % (d["n74"]["mean"], d["n74"]["lo"], d["n74"]["hi"], d["n71"]["mean"], d["n71"]["lo"], d["n71"]["hi"], d["Vmean"]))
vv = VAL.groupby("year")[["valid_dz", "valid_ring", "nobs_dz", "nobs_ring"]].mean(); vv.to_csv(f"{OUT}/valid_by_year.csv"); print(vv.round(3))
