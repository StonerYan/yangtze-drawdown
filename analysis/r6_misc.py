"""R6 - Land-cover comparability before impoundment (ESA CCI 2000), floodplain lakes 2018-2025, counterfactual scenarios, effect sizes, reservoirs with unknown commissioning year."""
import warnings; warnings.filterwarnings("ignore")
from common import *
rng = np.random.default_rng(8); out = {}
RS = pd.read_csv(f"{OUT}/res18.csv").set_index("reservoir"); WY = pd.read_parquet(f"{DC}/winyear.parquet")
# (1) ESA CCI 2000 composition (pre-impoundment for reservoirs commissioned after 2000)
pix = pd.read_parquet(f"{DC}/pixels.parquet"); att = pd.read_parquet(f"{DC}/pixel_attrs.parquet"); P = pix.merge(att, on="pid"); cy = pd.read_csv(f"{DC}/commission_year.csv").set_index("reservoir").commission
P["commission"] = P.reservoir.map(cy)
def grp(c):
    return np.select([np.isin(c, [10, 11, 12, 20, 30, 40]), np.isin(c, list(range(50, 101))), np.isin(c, [110, 120, 121, 122, 130]), np.isin(c, [160, 170, 180]), c == 190, np.isin(c, [140, 150, 151, 152, 153, 200, 201, 202]), c == 210],
                     ["cropland (incl. mosaics)", "tree cover", "shrub/grass", "wetland", "built-up", "bare/sparse", "water"], "other")
P["cci2000g"] = grp(P.cci2000.values); P["band"] = np.where(P.cls == "dz", "dz", pd.cut(P.dist, [59, 300, 600, 1100], labels=["60-300", "300-600", "600-1050"]).astype(str))
tab = {}
for lab, sub in (("all", P), ("commissioned_after_2000", P[P.commission > 2000])):
    t = pd.crosstab(sub.band, sub.cci2000g, normalize="index").round(3); tab[lab] = t.to_dict("index")
out["cci2000"] = tab; print(pd.crosstab(P[P.commission > 2000].band, P[P.commission > 2000].cci2000g, normalize="index").round(3))
# (2) lakes 2018-2025
wy = pd.read_parquet(f"{ROOT}/data/cee_wet/winyear.parquet") if os.path.exists(f"{ROOT}/data/cee_wet/winyear.parquet") else None
if wy is not None:
    cols = {"S": "shore", "R": "R_60_300", "G": "G_R_60_300"}; lk = {}
    ws = {}
    for k, c in cols.items():
        p = wy.pivot(index="window", columns="year", values=c).reindex(columns=range(2018, 2026)); ws[k] = pd.Series(ts_slope_rows(p.values, np.arange(2018, 2026, dtype=float)) * 1e3, index=p.index)
    ws = pd.DataFrame(ws); ws["lake"] = wy.drop_duplicates("window").set_index("window").reservoir.reindex(ws.index)
    sites = ws.groupby("lake")[["S", "R", "G"]].agg(["mean", "count"]); lkm = ws.groupby("lake")[["S", "R", "G"]].mean()
    out["lakes"] = {"n_win": int(ws.S.notna().sum()), "n_lake": int(len(lkm)), "mean": {v: boot_mean(lkm[v].values, B=5000) for v in ["S", "R", "G"]}, "by_lake": {}}
    for l, g in ws.groupby("lake"):
        out["lakes"]["by_lake"][l] = {v: dict(mean=float(g[v].mean()), lo=float(np.percentile(g[v].dropna().values[rng.integers(0, g[v].notna().sum(), (4000, g[v].notna().sum()))].mean(1), 2.5)),
                                              hi=float(np.percentile(g[v].dropna().values[rng.integers(0, g[v].notna().sum(), (4000, g[v].notna().sum()))].mean(1), 97.5)), n=int(g[v].notna().sum())) for v in ["S", "R", "G"]}
    out["lakes"]["n_R_pos"] = int((lkm.R > 0).sum()); print(json.dumps(out["lakes"]["mean"], default=float)); print(lkm.round(2))
    ws.to_csv(f"{OUT}/lake_window_slopes18.csv")
# (3) counterfactual scenarios: effect = b_S - lambda * b_R
sc = {}
for lab, idx in (("n74", RS.index), ("n71", RS.index[~RS.first_filling])):
    d = RS.loc[idx]; sc[lab] = {}
    for lam in (0, 0.25, 0.5, 0.75, 1.0):
        v = (d.S - lam * d.R).values; b = boot_mean(v, B=4000); sc[lab][str(lam)] = [b["mean"], b["lo"], b["hi"]]
out["scenarios"] = sc; print(sc)
# (4) effect sizes: cumulative change over 7 yearly steps relative to the mean level and to the interannual SD; levels from window-year medians
lv = WY[WY.year.between(2018, 2025)].groupby("reservoir")[["shore", "R_60_300"]].mean(); es = {}
for lab, idx in (("n74", RS.index), ("n71", RS.index[~RS.first_filling])):
    d = RS.loc[idx]; l = lv.loc[idx]; es[lab] = dict(S_level=float(l.shore.mean()), R_level=float(l.R_60_300.mean()), S_cum=float(d.S.mean() * 7e-3), R_cum=float(d.R.mean() * 7e-3), G_cum=float(d.G.mean() * 7e-3),
                                                    S_rel_pct=float(d.S.mean() * 7e-3 / l.shore.mean() * 100), R_rel_pct=float(d.R.mean() * 7e-3 / l.R_60_300.mean() * 100))
out["effect_size"] = es; print(es)
# (5) reservoirs with unknown commissioning year
unk = RS[RS.commission.isna()]; out["unknown_commission"] = dict(n=int(len(unk)), S=unk.S.round(2).to_dict(), R=unk.R.round(2).to_dict(), S_mean=float(unk.S.mean()))
print(out["unknown_commission"])
# (6) first-filling reservoirs: peak EVI2 of the drawdown mask by year
ff = WY[WY.reservoir.isin(FIRST)].groupby(["reservoir", "year"])[["shore", "R_60_300"]].mean().round(3); out["first_filling_series"] = {r: {int(y): [float(row.shore), float(row.R_60_300)] for y, row in ff.loc[r].iterrows()} for r in FIRST}
print(ff.unstack(0).round(2).to_string())
# (7) V (vegetated fraction) and S: sign agreement and rank correlation
VF = pd.read_parquet(f"{OUT}/vegfrac_window_slopes.parquet"); v = VF[(VF.thr == 0.2) & (VF.lo == 15) & (VF.hi > 95)].groupby("reservoir").V_slope.mean() * 100
j = RS[["S"]].join(v.rename("V")).dropna(); from scipy.stats import spearmanr
out["V_vs_S"] = dict(rho=float(spearmanr(j.S, j.V)[0]), sign_agree=float((np.sign(j.S) == np.sign(j.V)).mean()), n_V_neg=int((j.V < 0).sum()), n_S_neg=int((j.S < 0).sum()), n=int(len(j)))
print(out["V_vs_S"])
dump(out, "r6_misc")
