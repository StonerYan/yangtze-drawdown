"""R1 - Core estimates on the balanced 2018-2025 window (2017 has 6.6 clear looks per window on average and a valid S series in only 52% of windows).
 A. S, R, G trends for all 74 reservoirs, the 71 reservoirs in operation throughout, and the 3 first-filling reservoirs.
 B. Sensitivity: reference definitions, composites, trend windows, leave-one-year-out, year-block resampling.
 C. Clear-observation counts and density-adjusted trends.
 D. Regression adjustment on R; interannual coupling; null expectations for differencing.
Reservoir level = mean of window Theil-Sen slopes; units 1e-3 EVI2 per year unless noted."""
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import binomtest, spearmanr, pearsonr
from common import *
rng = np.random.default_rng(2026)
WY = pd.read_parquet(f"{DC}/winyear.parquet"); W = pd.read_parquet(f"{D2}/win_table.parquet")
wmeta = W.drop_duplicates("reservoir").set_index("reservoir")[["basin", "lon", "lat", "commission", "reach"]]
wmeta["grid3"] = np.floor(wmeta.lon / 3).astype(int) * 100 + np.floor(wmeta.lat / 3).astype(int)
VARS = {"S": "shore", "R": "R_60_300", "G": "G_R_60_300"}
def win_slopes(df, years, cols):
    out = {}
    for k, c in cols.items():
        p = df.pivot(index="window", columns="year", values=c).reindex(columns=years)
        out[k] = pd.Series(ts_slope_rows(p.values, np.asarray(years, float)), index=p.index)
    o = pd.DataFrame(out); o["reservoir"] = df.drop_duplicates("window").set_index("window").reservoir.reindex(o.index); return o
def res_slopes(df, years, cols):
    ws = win_slopes(df, years, cols); return ws.groupby("reservoir")[list(cols)].mean() * 1e3
Y18 = list(range(2018, 2026))
RS = res_slopes(WY, Y18, VARS).join(wmeta)
RS["first_filling"] = RS.index.isin(FIRST); RS.to_csv(f"{OUT}/res18.csv")
sets = {"n74": RS.index, "n71": RS.index[~RS.first_filling], "n3": RS.index[RS.first_filling]}
out = {"years": [2018, 2025]}
# ---------------- A. headline ----------------
A = {}
for k, idx in sets.items():
    d = RS.loc[idx]; a = {}
    for v in ["S", "R", "G"]:
        b = boot_mean(d[v].values); b["sub_basin_ci"] = block_boot_mean(d[v].values, d.basin.values) if k != "n3" else None
        b["grid3_ci"] = block_boot_mean(d[v].values, d.grid3.values) if k != "n3" else None; a[v] = b
    a["gap_R_minus_S"] = boot_mean((d.R - d.S).values)
    a["n_G_neg"] = int((d.G < 0).sum()); a["n_S_neg"] = int((d.S < 0).sum()); a["n_R_neg"] = int((d.R < 0).sum())
    a["n_G_neg_S_nonneg"] = int(((d.G < 0) & (d.S >= 0)).sum()); a["n"] = int(len(d))
    A[k] = a
out["A"] = A
# mean-of-windows vs additive check
out["G_vs_S_minus_R_74"] = [float(RS.G.mean()), float((RS.S - RS.R).mean())]
# pooled by sub-basin (random-effects, DerSimonian-Laird) for R, S, G among the 71
def dl(y, v):
    w = 1 / v; mu = (w * y).sum() / w.sum(); Q = (w * (y - mu) ** 2).sum(); k = len(y)
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); ws = 1 / (v + tau2); m = (ws * y).sum() / ws.sum(); se = np.sqrt(1 / ws.sum())
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se), float(tau2), float(max(0, (Q - (k - 1)) / Q) if Q > 0 else 0)
bsum = {}
d71 = RS.loc[sets["n71"]]
for v in ["S", "R", "G"]:
    g = d71.groupby("basin")[v].agg(["mean", "std", "count"]); g = g[g["count"] >= 2]; g["se2"] = (g["std"] ** 2 / g["count"]).clip(lower=1e-6)
    bsum[v] = dict(zip(["mu", "lo", "hi", "tau2", "I2"], dl(g["mean"].values, g["se2"].values)), k=int(len(g)))
    bsum[v + "_by_basin"] = g["mean"].round(3).to_dict()
out["subbasin_RE_71"] = bsum
# ---------------- B. sensitivity ----------------
B = {}
refs = {"R_60_300": "G_R_60_300", "R_300_600": "G_R_300_600", "R_600_1050": "G_R_600_1050", "R_slope": "G_R_slope", "R_stable": "G_R_stable", "R_natural": "G_R_natural", "R_regional": "G_R_regional"}
cols = {"S": "shore"}; [cols.update({k: k, "G_" + k: g}) for k, g in refs.items()]
rr = res_slopes(WY, Y18, cols)
B["refdefs"] = {k: {s: {v: boot_mean(rr.loc[idx, v].values, B=2000)["mean"] for v in (k, "G_" + k)} for s, idx in sets.items() if s != "n3"} for k in refs}
B["refdefs_ci"] = {k: {s: boot_mean(rr.loc[idx, k].values, B=2000) for s, idx in sets.items() if s != "n3"} for k in refs}
# windows
wins = {"2017-2025 (original)": list(range(2017, 2026)), "2018-2025": Y18, "2019-2025": list(range(2019, 2026)), "2018-2024": list(range(2018, 2025)), "2018-2023": list(range(2018, 2024))}
B["windows"] = {}
for lab, ys in wins.items():
    r = res_slopes(WY, ys, VARS); B["windows"][lab] = {s: {v: boot_mean(r.loc[idx, v].values, B=2000) for v in VARS} for s, idx in sets.items() if s != "n3"}
# leave-one-year-out
B["loyo"] = {}
for y in Y18:
    ys = [t for t in Y18 if t != y]; r = res_slopes(WY, ys, VARS); B["loyo"][y] = {s: {v: float(r.loc[idx, v].mean()) for v in VARS} for s, idx in sets.items() if s != "n3"}
for s in ("n74", "n71"):
    B["loyo_range_" + s] = {v: [float(min(B["loyo"][y][s][v] for y in Y18)), float(max(B["loyo"][y][s][v] for y in Y18))] for v in VARS}
# year-block bootstrap (years resampled with replacement; reservoirs fixed) and two-way bootstrap
P = {k: WY.pivot(index="window", columns="year", values=c).reindex(columns=Y18) for k, c in VARS.items()}
wres = WY.drop_duplicates("window").set_index("window").reservoir.reindex(P["S"].index)
ridx = pd.Index(RS.index); wi = ridx.get_indexer(wres.values); nres = len(ridx)
BB = 1000; ybs = {v: np.full((BB, nres), np.nan) for v in VARS}; yrs = np.array(Y18, float); done = 0
for b in range(BB):
    idx = rng.integers(0, 8, 8)
    if len(np.unique(idx)) < 3: continue
    for v in VARS:
        sl = ts_slope_rows(P[v].values[:, idx], yrs[idx])
        s = np.zeros(nres); c = np.zeros(nres); ok = np.isfinite(sl)
        np.add.at(s, wi[ok], sl[ok]); np.add.at(c, wi[ok], 1); ybs[v][b] = s / np.maximum(c, 1) * 1e3
    done += 1
yb = {}
for sname, idx in sets.items():
    if sname == "n3": continue
    pos = ridx.get_indexer(idx); yb[sname] = {}
    for v in VARS:
        m = np.nanmean(ybs[v][:, pos], 1); m = m[np.isfinite(m)]
        two = np.array([np.nanmean(ybs[v][b, rng.choice(pos, len(pos))]) for b in range(BB) if np.isfinite(ybs[v][b]).any()])
        yb[sname][v] = dict(year_ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], two_way_ci=[float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))])
B["year_block"] = yb; B["year_block_draws"] = done
out["B"] = B
# ---------------- C. clear-observation counts and density-adjusted trends ----------------
OD = pd.read_parquet(f"{D2}/obsdensity_winyear.parquet"); OD = OD[OD.window.isin(W.window)]
C = {"nobs_ring_by_year": OD.groupby("year").nobs_ring.mean().round(2).to_dict(), "nobs_shore_by_year": OD.groupby("year").nobs_shore.mean().round(2).to_dict()}
v17 = WY[WY.year == 2017]; C["valid_2017"] = dict(S=float(v17.shore.notna().mean()), R=float(v17.R_60_300.notna().mean()))
C["valid_2018_2025"] = dict(S=float(WY[WY.year >= 2018].shore.notna().mean()), R=float(WY[WY.year >= 2018].R_60_300.notna().mean()))
vf = pd.read_parquet(f"{OUT}/valid_by_year.csv".replace(".csv", ".csv")) if False else pd.read_csv(f"{OUT}/valid_by_year.csv")
C["valid_pixel_fraction"] = vf.set_index("year")[["valid_dz", "valid_ring"]].round(3).to_dict()
def fe_slope(df, ycol, ncol, resv, B=500, seed=5):
    d = df[["window", "reservoir", "year", ycol, ncol]].dropna().copy()
    def fit(dd, adj):
        X = dd[["year"] + ([ncol] if adj else [])].astype(float).copy(); X["year"] = X.year - 2018
        y = dd[ycol]; g = dd.window
        Xd = X - X.groupby(g).transform("mean"); yd = y - y.groupby(g).transform("mean")
        beta = np.linalg.lstsq(Xd.values, yd.values, rcond=None)[0]; return beta[0] * 1e3
    r = np.random.default_rng(seed); out_ = {}
    for adj in (False, True):
        pt = fit(d, adj); rv = d.reservoir.unique(); bs = []
        grp = {k: g for k, g in d.groupby("reservoir")}
        for _ in range(B):
            s = r.choice(rv, len(rv)); dd = pd.concat([grp[x].assign(window=grp[x].window + f"#{i}") for i, x in enumerate(s)]); bs.append(fit(dd, adj))
        out_["adj" if adj else "raw"] = [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out_
C["R_trend_density_adj_74"] = fe_slope(OD, "R_orig", "nobs_ring", None); C["S_trend_density_adj_74"] = fe_slope(OD, "shore_orig", "nobs_shore", None)
OD71 = OD[~OD.reservoir.isin(FIRST)]
C["R_trend_density_adj_71"] = fe_slope(OD71, "R_orig", "nobs_ring", None); C["S_trend_density_adj_71"] = fe_slope(OD71, "shore_orig", "nobs_shore", None)
# composites (2018-2025 window; all three computed on the same years)
ODw = OD.copy(); comp = {}
for lab, (sc, rc) in {"second-highest (primary)": ("shore_orig", "R_orig"), "fixed six looks": ("shore_fix6", "R_fix6"), "Jul-Sep mean": ("shore_julsep", "R_julsep")}.items():
    ODw["G_" + lab] = ODw[sc] - ODw[rc]
    r = res_slopes(ODw.rename(columns={sc: "S_", rc: "R_"}), Y18, {"S": "S_", "R": "R_", "G": "G_" + lab})
    comp[lab] = {s: {v: boot_mean(r.loc[idx.intersection(r.index), v].values, B=3000) for v in ["S", "R", "G"]} for s, idx in sets.items() if s != "n3"}
C["composites"] = comp
out["C"] = C
# ---------------- D. coupling, regression adjustment ----------------
RY = WY[WY.year.between(2018, 2025)].groupby(["reservoir", "year"])[["shore", "R_60_300"]].mean().reset_index()
RY["G"] = RY.shore - RY.R_60_300
pv = {k: RY.pivot(index="reservoir", columns="year", values=c)[Y18] for k, c in {"S": "shore", "R": "R_60_300", "G": "G"}.items()}
ok = pv["S"].notna().all(1) & pv["R"].notna().all(1); rok = pv["S"].index[ok]
M = {k: v.loc[rok].values * 1e3 for k, v in pv.items()}; x = np.arange(8) - 3.5; n = len(rok)
def detr(Mx):
    c = Mx - Mx.mean(1, keepdims=True); b = c @ x / (x @ x); return b, c - np.outer(b, x)
bS, eS = detr(M["S"]); bR, eR = detr(M["R"]); bG, eG = detr(M["G"])
D = {"n_complete": int(n), "n71": int(np.isin(rok, sets["n71"]).sum())}
for lab, mask in (("n_all", np.ones(n, bool)), ("n71", np.isin(rok, sets["n71"]))):
    a_, b_ = eS[mask], eR[mask]; rr_ = np.array([np.corrcoef(a_[i], b_[i])[0, 1] for i in range(mask.sum())])
    pooled = pearsonr(a_.ravel(), b_.ravel()); sdS, sdR, sdG = a_.std(), b_.std(), eG[mask].std()
    bt = binomtest(int((rr_ > 0).sum()), len(rr_), 0.5)
    # permutation null: shuffle years of R residuals independently within each reservoir
    nullm, nullp = [], []
    for _ in range(5000):
        rr0 = np.array([np.corrcoef(a_[i], b_[i][rng.permutation(8)])[0, 1] for i in range(len(rr_))]); nullm.append(np.median(rr0)); nullp.append((rr0 > 0).mean())
    ratio_b = []
    for _ in range(2000):
        i = rng.integers(0, mask.sum(), mask.sum()); ratio_b.append(eG[mask][i].std() / a_[i].std())
    D[lab] = dict(n=int(mask.sum()), r_pooled=float(pooled[0]), p_pooled=float(pooled[1]), r_med=float(np.median(rr_)), frac_pos=float((rr_ > 0).mean()), n_pos=int((rr_ > 0).sum()), sign_p=float(bt.pvalue),
                  null_median_ci=[float(np.percentile(nullm, 2.5)), float(np.percentile(nullm, 97.5))], null_frac_pos_ci=[float(np.percentile(nullp, 2.5)), float(np.percentile(nullp, 97.5))],
                  perm_p_median=float((np.array(nullm) >= np.median(rr_)).mean()), sdS=float(sdS), sdR=float(sdR), sdG=float(sdG), ratio=float(sdG / sdS), ratio_ci=[float(np.percentile(ratio_b, 2.5)), float(np.percentile(ratio_b, 97.5))],
                  ratio_expected_if_independent=float(np.sqrt(sdS ** 2 + sdR ** 2) / sdS), spearman_bS_bR=[float(spearmanr(bS[mask], bR[mask])[0]), float(spearmanr(bS[mask], bR[mask])[1])])
# regression adjustment: S_it = a_i + delta*t + beta*R_it  (reservoir FE; cluster bootstrap)
def adj_fit(Sm, Rm, with_trend=True, year_fe=False):
    n_, T = Sm.shape; t = np.tile(x, (n_, 1)); cols = []
    Sd = Sm - Sm.mean(1, keepdims=True); Rd = Rm - Rm.mean(1, keepdims=True)
    if year_fe:
        Sd = Sd - Sd.mean(0, keepdims=True); Rd = Rd - Rd.mean(0, keepdims=True); X = Rd.ravel()[:, None]
    else:
        X = np.column_stack([t.ravel(), Rd.ravel()])
    b = np.linalg.lstsq(X, Sd.ravel(), rcond=None)[0]; return b
RA = {}
for lab, mask in (("n74", np.ones(n, bool)), ("n71", np.isin(rok, sets["n71"]))):
    Sm, Rm = M["S"][mask], M["R"][mask]; pt = adj_fit(Sm, Rm); pt2 = adj_fit(Sm, Rm, year_fe=True)
    # OLS naive trend of S alone for comparison
    naive = (Sm - Sm.mean(1, keepdims=True)) @ x / (x @ x)
    bs, bs2 = [], []
    for _ in range(3000):
        i = rng.integers(0, mask.sum(), mask.sum()); bs.append(adj_fit(Sm[i], Rm[i])); bs2.append(adj_fit(Sm[i], Rm[i], year_fe=True)[0])
    bs = np.array(bs); q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    bR_ols = ((Rm - Rm.mean(1, keepdims=True)) @ x / (x @ x)).mean()
    RA[lab] = dict(delta=float(pt[0]), delta_ci=q(bs[:, 0]), beta=float(pt[1]), beta_ci=q(bs[:, 1]), beta_yearFE=float(pt2[0]), beta_yearFE_ci=q(bs2),
                   naive_S_trend_ols=float(naive.mean()), R_trend_ols=float(bR_ols), implied_unit_beta_G=float(naive.mean() - bR_ols), n=int(mask.sum()))
D["regression_adjustment"] = RA
# annual anomalies and 2022
anom = {}
for k in M:
    A_ = M[k] - M[k].mean(1, keepdims=True)
    bs = np.array([A_[rng.integers(0, n, n)].mean(0) for _ in range(3000)])
    anom[k] = {int(y): [float(A_[:, j].mean()), float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))] for j, y in enumerate(Y18)}
D["annual_anomalies"] = anom
pd.DataFrame([dict(var=k, year=y, mean=v[0], lo=v[1], hi=v[2]) for k, d in anom.items() for y, v in d.items()]).to_csv(f"{OUT}/ts_anom18.csv", index=False)
# detrended residual series (for figure)
pd.DataFrame({"reservoir": np.repeat(rok, 8), "year": np.tile(Y18, n), "eS": eS.ravel(), "eR": eR.ravel(), "eG": eG.ravel()}).to_csv(f"{OUT}/detrended18.csv", index=False)
out["D"] = D
dump(out, "r1_core")
print(json.dumps({k: out["A"][k] if False else {kk: (vv if not isinstance(vv, dict) else {a: (round(b, 3) if isinstance(b, float) else b) for a, b in vv.items() if a in ("mean", "lo", "hi", "median", "sub_basin_ci")}) for kk, vv in out["A"][k].items()} for k in out["A"]}, indent=1, default=float))
print("loyo range", B["loyo_range_n74"], B["loyo_range_n71"]); print("year block", json.dumps(yb)); print("density", json.dumps({k: v for k, v in C.items() if "density" in k}))
print("composites", {k: {s: {v: round(x_["mean"], 2) for v, x_ in d_.items()} for s, d_ in vv.items()} for k, vv in comp.items()})
print("coupling", json.dumps({k: D[k] for k in ("n_all", "n71")}, default=float)); print("regadj", json.dumps(RA)); print("anom2022", anom["S"][2022], anom["R"][2022], anom["G"][2022])
print({k: {s: {v: round(B["windows"][k][s][v]["mean"], 2) for v in VARS} for s in B["windows"][k]} for k in B["windows"]})
print("refdefs", json.dumps(B["refdefs"]))
