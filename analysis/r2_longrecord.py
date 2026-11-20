"""R2 - Long records: (a) valid Landsat looks per year; (b) sensor-offset (step) model and formal break test for the upland series;
(c) exclusion of 2011-2014; (d) before-after contrast around impoundment for reservoirs commissioned 2007-2015 (pixels were land before filling);
(e) MODIS 2018-2025; (f) Sentinel-2 vs Landsat OLI window trends 2018-2025."""
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import theilslopes, pearsonr, spearmanr
from common import *
rng = np.random.default_rng(77)
L = pd.read_parquet(f"{D2}/landsat_density_winyear.parquet"); out = {}
# (a) looks per year
al = L[L.set == "all"]; out["looks_by_year"] = al.groupby("year").nlooks_ring.mean().round(2).to_dict(); out["windows_by_year"] = al.groupby("year").window.nunique().to_dict()
out["looks_by_year_sensorfamily"] = {s: L[L.set == s].groupby("year").nlooks_ring.mean().round(2).to_dict() for s in ("TMETM", "ETM", "OLI")}
# (b) models on window-FE panel
def demean(d, cols): return d[cols] - d.groupby("window")[cols].transform("mean")
def fit(d, v, kind, tau=None):
    t = d.year.values - 2000.; X = {"t": t}
    if kind in ("step", "hinge_step"): X["step"] = (d.year.values >= 2013).astype(float)
    if kind in ("hinge", "hinge_step"): X["h"] = np.maximum(0, d.year.values - tau)
    df = pd.DataFrame(X); df["y"] = d[v].values; df["window"] = d.window.values
    dm = df[list(X) + ["y"]] - df.groupby("window")[list(X) + ["y"]].transform("mean")
    beta = np.linalg.lstsq(dm[list(X)].values, dm.y.values, rcond=None)[0]; r = dm.y.values - dm[list(X)].values @ beta
    return dict(zip(X, beta)), float(r @ r)
TAUS = np.arange(2005, 2020.01, 0.5)
def best_hinge(d, v, kind="hinge"):
    best = min(((fit(d, v, kind, tau)[1], tau) for tau in TAUS)); b, sse = fit(d, v, kind, best[1]); return best[1], b, sse
def bic(sse, n, k): return n * np.log(sse / n) + k * np.log(n)
def prep(sel, v, y0=2000, y1=2025):
    d = L[(L.set == sel) & L.year.between(y0, y1)][["window", "reservoir", "year", v]].dropna(); return d[d.groupby("window").year.transform("size") >= 8]
def models(sel, v, y0=2000, y1=2025, B=300):
    d = prep(sel, v, y0, y1); n = len(d); kw = d.window.nunique(); o = {"n_win": int(kw), "n_obs": int(n)}
    bl, sl = fit(d, v, "linear"); o["linear"] = float(bl["t"] * 1e3)
    tau, bh, sh = best_hinge(d, v); o["hinge"] = dict(tau=float(tau), pre=float(bh["t"] * 1e3), post=float((bh["t"] + bh["h"]) * 1e3))
    bs_, ss_ = fit(d, v, "step"); o["step"] = dict(slope=float(bs_["t"] * 1e3), step=float(bs_["step"] * 1e3))
    tau2, bhs, shs = best_hinge(d, v, "hinge_step"); o["hinge_step"] = dict(tau=float(tau2), pre=float(bhs["t"] * 1e3), post=float((bhs["t"] + bhs["h"]) * 1e3), step=float(bhs["step"] * 1e3))
    o["BIC"] = dict(linear=bic(sl, n, kw + 1), step=bic(ss_, n, kw + 2), hinge=bic(sh, n, kw + 3), hinge_step=bic(shs, n, kw + 4))
    # supF-type statistic and residual-vector bootstrap p-value for hinge vs linear
    F_obs = (sl - sh) / (sh / (n - kw - 2)); o["supF"] = float(F_obs)
    d = d.assign(res=0.0); b0, _ = fit(d, v, "linear")
    t = d.year.values - 2000.; dm = demean(d.assign(t=t), [v, "t"]); d["fit_lin"] = d.groupby("window")[v].transform("mean") + b0["t"] * (t - d.assign(t=t).groupby("window").t.transform("mean")); d["res"] = d[v] - d.fit_lin
    piv = d.pivot(index="window", columns="year", values="res"); ws = piv.index.values; cnt = 0; nb = 200
    for _ in range(nb):
        perm = rng.permutation(len(ws)); yb = d.copy()
        newres = []
        for w, w2 in zip(ws, ws[perm]):
            sub = d[d.window == w]; r2 = piv.loc[w2].reindex(sub.year.values).values; r2 = np.where(np.isfinite(r2), r2, rng.choice(piv.loc[w2].dropna().values, len(sub)))
            newres.append(pd.Series(r2, index=sub.index))
        yb["y_b"] = d.fit_lin + pd.concat(newres).reindex(d.index)
        bl_, sl_ = fit(yb, "y_b", "linear"); _, _, sh_ = best_hinge(yb, "y_b"); cnt += ((sl_ - sh_) / (sh_ / (n - kw - 2))) >= F_obs
    o["supF_boot_p"] = float((cnt + 1) / (nb + 1))
    # cluster bootstrap of step model slope/step and hinge-step post slope
    rv = d.reservoir.unique(); grp = {k: g for k, g in d.groupby("reservoir")}; bst = []
    for _ in range(B):
        s = rng.choice(rv, len(rv)); dd = pd.concat([grp[x].assign(window=grp[x].window + f"#{i}") for i, x in enumerate(s)]); q, _ = fit(dd, v, "step"); bst.append([q["t"] * 1e3, q["step"] * 1e3])
    bst = np.array(bst); o["step_ci"] = dict(slope=[float(np.percentile(bst[:, 0], 2.5)), float(np.percentile(bst[:, 0], 97.5))], step=[float(np.percentile(bst[:, 1], 2.5)), float(np.percentile(bst[:, 1], 97.5))])
    return o
out["models"] = {}
for sel, v, y0, y1 in [("all", "R_fix3", 2000, 2025), ("harm", "R_fix3", 2000, 2025), ("harm", "R_peak2", 2000, 2025), ("ETM", "R_fix3", 2000, 2019)]:
    out["models"][f"{sel}:{v}:{y0}-{y1}"] = models(sel, v, y0, y1); print(sel, v, json.dumps(out["models"][f"{sel}:{v}:{y0}-{y1}"], default=float)[:900], flush=True)
# (c) exclude 2011-2014: separate linear trends before and after
def gap_fit(sel, v, B=500):
    d = prep(sel, v)
    def f(dd):
        res = {}
        for lab, m in (("pre_2000_2010", dd.year <= 2010), ("post_2015_2025", dd.year >= 2015)):
            x = dd[m]; res[lab] = fit(x, v, "linear")[0]["t"] * 1e3
        return res
    pt = f(d); rv = d.reservoir.unique(); grp = {k: g for k, g in d.groupby("reservoir")}; bs = []
    for _ in range(B):
        s = rng.choice(rv, len(rv)); dd = pd.concat([grp[x].assign(window=grp[x].window + f"#{i}") for i, x in enumerate(s)]); bs.append(f(dd))
    bs = pd.DataFrame(bs); o = {k: [float(pt[k]), float(bs[k].quantile(.025)), float(bs[k].quantile(.975))] for k in pt}
    o["diff"] = [float(pt["post_2015_2025"] - pt["pre_2000_2010"]), float((bs.post_2015_2025 - bs.pre_2000_2010).quantile(.025)), float((bs.post_2015_2025 - bs.pre_2000_2010).quantile(.975))]; return o
out["exclude_2011_2014"] = {f"{s}:{v}": gap_fit(s, v) for s, v in [("harm", "R_fix3"), ("harm", "R_peak2"), ("all", "R_fix3")]}; print(out["exclude_2011_2014"])
# single sensor families (from the original analysis, recomputed here for completeness)
sp = {}
for sel, v, y0, y1 in [("ETM", "R_fix3", 2000, 2012), ("ETM", "R_fix3", 2013, 2019), ("OLI", "R_fix3", 2013, 2025), ("OLI", "R_fix3", 2018, 2025), ("TMETM", "R_fix3", 2000, 2012)]:
    s = []
    for w, g in L[(L.set == sel) & L.year.between(y0, y1)][["window", "year", v]].dropna().groupby("window"):
        if len(g) >= 5: s.append(theilslopes(g[v], g.year)[0] * 1e3)
    s = np.array(s); b = rng.choice(s, (4000, len(s))).mean(1); sp[f"{sel}:{y0}-{y1}"] = dict(mean=float(s.mean()), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)), n=int(len(s)))
out["single_sensor"] = sp; print(sp)
# (d) before-after around impoundment (harmonised fixed-density S, R); commissioning 2007-2015, excluding Three Gorges (impounded in stages from 2003)
h = L[L.set == "harm"].copy(); h = h[h.commission.between(2007, 2015) & (h.reservoir != "Mainstem_Three_Gorges")]
rows = []
for r, g in h.groupby("reservoir"):
    c = g.commission.iat[0]; pre = g[g.year <= c - 2]; post = g[g.year >= c + 2]; d = dict(reservoir=r, commission=c, n_pre=int(pre.S_fix3.notna().sum()), n_post=int(post.S_fix3.notna().sum()))
    for v in ("S_fix3", "R_fix3"):
        a = pre[["year", v]].dropna(); b = post[["year", v]].dropna()
        d["pre_" + v] = theilslopes(a[v], a.year)[0] * 1e3 if len(a) >= 5 else np.nan; d["post_" + v] = theilslopes(b[v], b.year)[0] * 1e3 if len(b) >= 5 else np.nan
        d["lvl_pre_" + v] = a[v].mean(); d["lvl_post_" + v] = b[v].mean()
    rows.append(d)
X = pd.DataFrame(rows); X["pre_diff"] = X.pre_S_fix3 - X.pre_R_fix3; X["post_diff"] = X.post_S_fix3 - X.post_R_fix3
X["lvl_step_S"] = X.lvl_post_S_fix3 - X.lvl_pre_S_fix3; X["lvl_step_R"] = X.lvl_post_R_fix3 - X.lvl_pre_R_fix3; X["lvl_did"] = X.lvl_step_S - X.lvl_step_R
X.to_csv(f"{OUT}/baci_landsat.csv", index=False)
out["baci"] = {c: boot_mean(X[c].dropna().values, B=5000) for c in ["pre_S_fix3", "pre_R_fix3", "pre_diff", "post_S_fix3", "post_R_fix3", "post_diff", "lvl_step_S", "lvl_step_R", "lvl_did"]}
out["baci"]["n_res"] = int(len(X)); print({k: (round(v["mean"], 3), round(v["lo"], 3), round(v["hi"], 3), v["n"]) for k, v in out["baci"].items() if k != "n_res"})
# (e) MODIS
ms = pd.read_parquet(f"{D2}/modis_res_series.parquet"); ms = ms[ms.system == "reservoir"]; mo = {}
for var in ("evi", "ndvi"):
    for lab, (y0, y1) in {"2001-2012": (2001, 2012), "2013-2019": (2013, 2019), "2018-2025": (2018, 2025), "2017-2025": (2017, 2025)}.items():
        s = []
        for r, g in ms[ms.year.between(y0, y1)][["reservoir", "year", var]].dropna().groupby("reservoir"):
            if len(g) >= 5: s.append(theilslopes(g[var], g.year)[0] * 1e3)
        s = np.array(s); mo[f"{var}:{lab}"] = boot_mean(s, B=4000)
out["modis"] = mo; print({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2), v["n"]) for k, v in mo.items()})
print("MODIS years available", sorted(ms.year.unique())[-3:])
# MODIS vs Sentinel-2 rank agreement, 2018-2025
RS = pd.read_csv(f"{OUT}/res18.csv").set_index("reservoir")
me = []
for r, g in ms[ms.year.between(2018, 2025)].groupby("reservoir"):
    if len(g) >= 5: me.append((r, theilslopes(g.evi, g.year)[0] * 1e3))
me = pd.DataFrame(me, columns=["reservoir", "modis"]).set_index("reservoir").join(RS[["R"]]).dropna(); rho = spearmanr(me.modis, me.R)
bs = [spearmanr(me.modis.values[i], me.R.values[i])[0] for i in (rng.integers(0, len(me), len(me)) for _ in range(2000))]
out["modis_vs_s2_R"] = dict(rho=float(rho[0]), p=float(rho[1]), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(len(me)), mean_modis=float(me.modis.mean()), mean_s2=float(me.R.mean()))
print(out["modis_vs_s2_R"])
# (f) Sentinel-2 vs Landsat OLI, 2018-2025 window slopes
WY = pd.read_parquet(f"{DC}/winyear.parquet"); lw = pd.read_parquet(f"{DC}/results/landsat_winyear.parquet"); rows = []
for w, g in lw.groupby("window"):
    g = g[g.year.between(2018, 2025)][["year", "peak_OLI_R"]].dropna(); s2 = WY[(WY.window == w) & WY.year.between(2018, 2025)][["year", "R_60_300"]].dropna()
    if len(g) >= 6 and len(s2) >= 6: rows.append(dict(window=w, landsat=theilslopes(g.peak_OLI_R, g.year)[0] * 1e3, s2=theilslopes(s2.R_60_300, s2.year)[0] * 1e3))
LS = pd.DataFrame(rows); r = pearsonr(LS.landsat, LS.s2)[0]; bs = [pearsonr(LS.landsat.values[i], LS.s2.values[i])[0] for i in (rng.integers(0, len(LS), len(LS)) for _ in range(2000))]
out["s2_vs_landsat_2018_2025"] = dict(n=int(len(LS)), r=float(r), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], mean_landsat=float(LS.landsat.mean()), mean_s2=float(LS.s2.mean()))
print(out["s2_vs_landsat_2018_2025"]); LS.to_csv(f"{OUT}/s2_landsat_pairs18.csv", index=False)
dump(out, "r2_longrecord")
