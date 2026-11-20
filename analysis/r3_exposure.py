"""R3 - Time-varying exposure and drawdown-zone vegetation within pixels (all reservoirs with daily storage: 52 reservoirs, 464 windows).
Exposure of pixel p on day t: storage V(t) < Q_V(1 - F_p/100), where F_p is JRC surface-water occurrence (%) and Q_V the quantile of the reservoir's daily storage record
(rank matching of occurrence to storage). Growing season = 15 Mar-30 Nov (the Sentinel-2 season). Response = annual peak EVI2 (second-highest clear value, >=6 clear looks).
Models: pixel FE + year FE (A); pixel FE + reservoir-by-year FE (B); standard errors clustered by reservoir.  Plus validation against Sentinel-2 for all windows."""
import glob, warnings; warnings.filterwarnings("ignore")
from common import *
from scipy.stats import theilslopes, pearsonr
pix = pd.read_parquet(f"{DC}/pixels.parquet"); wn = pd.read_csv(f"{DC}/windows.csv")
w52 = wn[wn.source == "52daily"]; pix = pix[pix.window.isin(w52.window) & (pix.cls == "dz")]
def storage(res):
    s = pd.read_csv(f"{ROOT}/data/pilot/storage_52_csv/{res}.csv", parse_dates=["date"]).set_index("date")["V_rs (km3)"]
    return s.reindex(pd.date_range(s.index.min(), s.index.max())).interpolate(limit=10)
rows = []; val = []; ST = {}
for res, gw in pix.groupby("reservoir"):
    V = storage(res); ST[res] = V; Vv = V.dropna().values
    amp = {}
    for y in range(2018, 2026):
        vy = V.reindex(pd.date_range(f"{y}-01-01", f"{y}-12-31"))
        amp[y] = (vy.max() - vy.min()) / max(Vv.max() - Vv.min(), 1e-9) if vy.notna().sum() >= 330 else np.nan
    for w, g in gw.groupby("window"):
        z = np.load(f"{DC}/s2/{w}.npz"); pid = z["pid"]; ii = np.isin(pid, g.pid.values); E = z["evi2"].astype(float)[ii]; pids = pid[ii]; dates = pd.to_datetime(z["dates"])
        F = g.set_index("pid").F.reindex(pids).values; q = np.nanquantile(Vv, np.clip(1 - F / 100, 0, 1))
        # validation on clear looks with storage
        v = V.reindex(dates).values; ok = np.isfinite(v); wet = v[None, ok] >= q[:, None]; obs = E[:, ok]; m = np.isfinite(obs); ow = obs < 0.05
        tp = (wet & ow & m).sum(); tn = (~wet & ~ow & m).sum(); fp = (wet & ~ow & m).sum(); fn_ = (~wet & ow & m).sum()
        if m.sum() > 0: val.append(dict(window=w, reservoir=res, n_obs=int(m.sum()), agree=(tp + tn) / m.sum(), bal_acc=0.5 * (tp / max(tp + fn_, 1) + tn / max(tn + fp, 1))))
        pk = pd.read_parquet(f"{DC}/peaks/{w}.parquet").pivot(index="pid", columns="year", values="peak").reindex(pids)
        for y in range(2018, 2026):
            days = pd.date_range(f"{y}-03-15", f"{y}-11-30"); vy = V.reindex(days)
            if V.reindex(pd.date_range(f"{y}-01-01", f"{y}-12-31")).notna().sum() < 330: continue
            vy = vy.ffill().bfill().values; ex = vy[None, :] < q[:, None]; early = days <= pd.Timestamp(f"{y}-06-30")
            pkv = pk[y].values
            for i in np.nonzero(np.isfinite(pkv))[0]:
                rows.append((res, w, pids[i], F[i], y, pkv[i], ex[i].mean(), ex[i][early].mean(), ex[i][~early].mean(), amp[y]))
    print(res, len(rows), flush=True)
PN = pd.DataFrame(rows, columns=["reservoir", "window", "pid", "F", "year", "peak", "exp_gs", "exp_early", "exp_late", "amp"])
PN.to_parquet(f"{OUT}/exposure_panel.parquet"); VAL = pd.DataFrame(val); VAL.to_csv(f"{OUT}/exposure_validation_all.csv", index=False)
out = {"validation": dict(n_windows=int(len(VAL)), n_res=int(VAL.reservoir.nunique()), agree_median=float(VAL.agree.median()), agree_iqr=[float(VAL.agree.quantile(.25)), float(VAL.agree.quantile(.75))],
       bal_median=float(VAL.bal_acc.median()), bal_iqr=[float(VAL.bal_acc.quantile(.25)), float(VAL.bal_acc.quantile(.75))], bal_min=float(VAL.bal_acc.min()), frac_bal_gt_07=float((VAL.bal_acc > 0.7).mean()))}
print(out["validation"]); print(PN.shape, PN.reservoir.nunique(), PN.pid.nunique())
# ---------------- regression ----------------
def two_way_demean(df, cols, fes, it=30):
    X = df[cols].copy().astype(float)
    for _ in range(it):
        for fe in fes: X = X - X.groupby(df[fe].values).transform("mean")
    return X
def cluster_ols(df, ycol, xcols, fes, cl):
    cols = [ycol] + xcols; Xd = two_way_demean(df, cols, fes); y = Xd[ycol].values; X = Xd[xcols].values
    XtX = X.T @ X; b = np.linalg.solve(XtX, X.T @ y); e = y - X @ b
    meat = np.zeros((len(xcols), len(xcols)))
    for k, idx in pd.Series(np.arange(len(df))).groupby(df[cl].values):
        s = (X[idx.values] * e[idx.values][:, None]).sum(0); meat += np.outer(s, s)
    G = df[cl].nunique(); n = len(df); V = np.linalg.inv(XtX) @ meat @ np.linalg.inv(XtX) * G / (G - 1)
    se = np.sqrt(np.diag(V)); return b, se, G, n
PN["ry"] = PN.reservoir + "_" + PN.year.astype(str); PN["uid"] = PN.window + "_" + PN.pid.astype(str)
main = PN[~PN.reservoir.isin(FIRST)].copy()
# keep pixels observed in >=3 years
main = main[main.groupby("uid").year.transform("size") >= 3]
res = {}
def rep(tag, df, xcols, fes):
    b, se, G, n = cluster_ols(df, "peak", xcols, fes, "reservoir"); res[tag] = {c: dict(beta=float(bb), se=float(s), lo=float(bb - 1.96 * s), hi=float(bb + 1.96 * s)) for c, bb, s in zip(xcols, b, se)}
    res[tag]["n_pixel_years"] = int(n); res[tag]["n_clusters"] = int(G); res[tag]["n_pixels"] = int(df.uid.nunique()); print(tag, {c: (round(bb, 4), round(s, 4)) for c, bb, s in zip(xcols, b, se)}, n, G, flush=True)
rep("A_pixel_year", main, ["exp_gs"], ["uid", "year"])
rep("A_plus_amp", main.dropna(subset=["amp"]), ["exp_gs", "amp"], ["uid", "year"])
rep("B_pixel_resyear", main, ["exp_gs"], ["uid", "ry"])
rep("A_early_late", main, ["exp_early", "exp_late"], ["uid", "year"])
rep("B_early_late", main, ["exp_early", "exp_late"], ["uid", "ry"])
rep("A_pixel_only", main, ["exp_gs"], ["uid"])
rep("A_all_incl_first", PN[PN.groupby("uid").year.transform("size") >= 3], ["exp_gs"], ["uid", "year"])
for lo, hi in [(15, 35), (35, 55), (55, 75), (75, 95.01)]:
    rep(f"A_F{lo}_{int(hi)}", main[(main.F >= lo) & (main.F < hi)], ["exp_gs"], ["uid", "year"])
# correlation of early and late exposure after removing pixel and year effects
Xd = two_way_demean(main, ["exp_early", "exp_late", "exp_gs"], ["uid", "year"]); res["corr_early_late_within"] = float(np.corrcoef(Xd.exp_early, Xd.exp_late)[0, 1])
Xd2 = two_way_demean(main, ["exp_early", "exp_late"], ["uid", "ry"]); res["corr_early_late_within_resyear"] = float(np.corrcoef(Xd2.exp_early, Xd2.exp_late)[0, 1])
res["sd_exp_gs_raw"] = float(main.exp_gs.std()); res["sd_exp_gs_within_pixel_year"] = float(Xd.exp_gs.std()); res["mean_exp_gs"] = float(main.exp_gs.mean())
out["reg"] = res
# ---------------- accounting: does exposure change explain the drawdown-zone trend? ----------------
bA = res["A_pixel_year"]["exp_gs"]["beta"]; acc = []
for r, g in main.groupby("reservoir"):
    ry = g.groupby("year")[["peak", "exp_gs"]].mean()
    if len(ry) >= 5:
        acc.append(dict(reservoir=r, n_years=len(ry), obs=theilslopes(ry.peak, ry.index)[0] * 1e3, d_exp=theilslopes(ry.exp_gs, ry.index)[0], pred=bA * theilslopes(ry.exp_gs, ry.index)[0] * 1e3))
ACC = pd.DataFrame(acc); ACC.to_csv(f"{OUT}/exposure_accounting.csv", index=False)
out["accounting"] = dict(n_res=int(len(ACC)), mean_obs=float(ACC.obs.mean()), mean_pred=float(ACC.pred.mean()), mean_d_exp_per_yr=float(ACC.d_exp.mean()), r=float(pearsonr(ACC.obs, ACC.pred)[0]), p=float(pearsonr(ACC.obs, ACC.pred)[1]),
                         share_obs_neg=float((ACC.obs < 0).mean()), sd_d_exp=float(ACC.d_exp.std()))
print(out["accounting"])
dump(out, "r3_exposure")
