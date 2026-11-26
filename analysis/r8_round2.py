"""R8 - Second-round additions (nothing here changes the model).
 A. Year-resampling intervals for the whole counterfactual bracket (lambda = 0, 0.5, 1) and for the gap R - S.
 B. Robustness of the across-reservoir association between modelled exposure trends and observed EVI2 trends.
 C. Change in the drawdown-zone trend across commissioning (18 reservoirs with pre-impoundment records).
 D. Matched-window comparison of Sentinel-2, Landsat and MODIS upland trends (2018-2024).
Units 1e-3 EVI2 per year unless noted."""
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr
from common import *

WY = pd.read_parquet(f"{DC}/winyear.parquet"); W = pd.read_parquet(f"{D2}/win_table.parquet")
wmeta = W.drop_duplicates("reservoir").set_index("reservoir")[["basin", "lon", "lat"]]
Y18 = list(range(2018, 2026)); yrs = np.array(Y18, float)
P = {k: WY.pivot(index="window", columns="year", values=c).reindex(columns=Y18) for k, c in {"S": "shore", "R": "R_60_300"}.items()}
wres = WY.drop_duplicates("window").set_index("window").reservoir.reindex(P["S"].index)
ridx = pd.Index(sorted(wres.unique())); wi = ridx.get_indexer(wres.values); nres = len(ridx)
def res_mean(sl):
    s = np.zeros(nres); c = np.zeros(nres); ok = np.isfinite(sl); np.add.at(s, wi[ok], sl[ok]); np.add.at(c, wi[ok], 1)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan) * 1e3
base = {k: res_mean(ts_slope_rows(P[k].values, yrs)) for k in P}
first = ridx.isin(FIRST); n71 = ~first
out = {}
rng = np.random.default_rng(2026)   # same draws as the year-block bootstrap in r1_core.py
BB = 1000; eff = {lam: [] for lam in (0.0, 0.5, 1.0)}; gap = []; Rb = []; Sb = []
for b in range(BB):
    idx = rng.integers(0, 8, 8)
    if len(np.unique(idx)) < 3: continue
    s = res_mean(ts_slope_rows(P["S"].values[:, idx], yrs[idx])); r = res_mean(ts_slope_rows(P["R"].values[:, idx], yrs[idx]))
    Sm = np.nanmean(s[n71]); Rm = np.nanmean(r[n71]); Sb.append(Sm); Rb.append(Rm)
    for lam in eff: eff[lam].append(Sm - lam * Rm)
    gap.append(Rm - Sm)
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
out["A"] = {"n71": {"lam0": ci(eff[0.0]), "lam05": ci(eff[0.5]), "lam1_diff": ci(eff[1.0]), "gap_R_minus_S": ci(gap), "R_year_ci": ci(Rb), "S_year_ci": ci(Sb),
                    "point": {"S": float(np.nanmean(base["S"][n71])), "R": float(np.nanmean(base["R"][n71]))}, "draws": len(gap)}}
# P(S - R < S) etc.: share of year-resamples in which G < S (R > 0) and in which R > 0
out["A"]["share_R_positive"] = float((np.array(Rb) > 0).mean()); out["A"]["share_S_negative"] = float((np.array(Sb) < 0).mean())

# ---- B. exposure accounting
ea = pd.read_csv(f"{OUT}/exposure_accounting.csv"); ea["basin"] = wmeta.basin.reindex(ea.reservoir).values
B = {"n": int(len(ea)), "pearson": list(map(float, pearsonr(ea.pred, ea.obs))), "spearman": list(map(float, spearmanr(ea.pred, ea.obs)))}
tr = ea[(ea.obs > ea.obs.quantile(0.04)) & (ea.pred > ea.pred.quantile(0.04))]
B["pearson_drop_extremes"] = [int(len(tr))] + list(map(float, pearsonr(tr.pred, tr.obs)))
dm = ea.copy()
for c in ("pred", "obs"): dm[c] = dm[c] - dm.groupby("basin")[c].transform("mean")
B["pearson_within_subbasin"] = list(map(float, pearsonr(dm.pred, dm.obs)))
B["n_basins"] = int(ea.basin.nunique())
B["slope_obs_on_exposure_trend"] = float(np.polyfit(ea.d_exp, ea.obs, 1)[0])
bs = []
for _ in range(5000):
    i = rng.integers(0, len(ea), len(ea)); bs.append(pearsonr(ea.pred.values[i], ea.obs.values[i])[0])
B["pearson_boot_ci"] = ci(bs)
out["B"] = B

# ---- C. commissioning
bl = pd.read_csv(f"{OUT}/baci_landsat.csv"); both = bl.dropna(subset=["pre_S_fix3", "post_S_fix3"])
d = (both.post_S_fix3 - both.pre_S_fix3).values
bm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
dR = (both.post_R_fix3 - both.pre_R_fix3).values; bR = [dR[rng.integers(0, len(dR), len(dR))].mean() for _ in range(5000)]
out["C"] = {"n": int(len(both)), "dS": [float(d.mean())] + ci(bm), "dR": [float(dR.mean())] + ci(bR),
            "pre_S": [float(bl.pre_S_fix3.mean()), int(bl.pre_S_fix3.notna().sum())], "post_S": [float(both.post_S_fix3.mean()), len(both)],
            "n_pre_years_median": float(bl.n_pre.median())}

# ---- D. matched windows
r1 = json.load(open(f"{OUT}/r1_core.json", encoding="utf-8"))
w = r1["B"]["windows"]
out["D"] = {"S2_2018_2024_n71_R": {k: w["2018-2024"]["n71"]["R"][k] for k in ("mean", "lo", "hi")}}
dump(out, "r8_round2"); print(json.dumps(out, indent=1, default=float))
