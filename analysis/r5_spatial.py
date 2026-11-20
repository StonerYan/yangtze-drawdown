"""R5 - Spatial structure, nested sampling and sampling frame.
 (a) Moran's I of reservoir trends (2018-2025).  (b) Nested multilevel model (window in reservoir in sub-basin) for S, R, G.
 (c) Site attributes versus the upland trend: reservoir-grouped random-forest skill and split-half reliability; rank correlations with climate/land trends.
 (d) Sub-basin means and climate trends.  (e) The 74 reservoirs against the 441-reservoir inventory."""
import warnings; warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, chi2_contingency, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from common import *
rng = np.random.default_rng(31); out = {}
WY = pd.read_parquet(f"{DC}/winyear.parquet"); W = pd.read_parquet(f"{D2}/win_table.parquet"); RS = pd.read_csv(f"{OUT}/res18.csv").set_index("reservoir")
Y18 = list(range(2018, 2026))
def win_slopes(df, years, cols):
    out_ = {}
    for k, c in cols.items():
        p = df.pivot(index="window", columns="year", values=c).reindex(columns=years); out_[k] = pd.Series(ts_slope_rows(p.values, np.asarray(years, float)) * 1e3, index=p.index)
    return pd.DataFrame(out_)
WS = win_slopes(WY, Y18, {"S": "shore", "R": "R_60_300", "G": "G_R_60_300"}); WS = WS.join(W.set_index("window").drop(columns=["shore", "R_60_300", "G_R_60_300"], errors="ignore"))
# (a) Moran's I
def moran(v, lon, lat, scheme, nperm=2000):
    v = np.asarray(v, float); ok = np.isfinite(v); v, lon, lat = v[ok], lon[ok], lat[ok]; n = len(v)
    la1, lo1 = np.radians(lat)[:, None], np.radians(lon)[:, None]; d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la1 - la1.T) / 2) ** 2 + np.cos(la1) * np.cos(la1.T) * np.sin((lo1 - lo1.T) / 2) ** 2))
    if scheme == "invdist300": Wm = np.where((d > 0) & (d <= 300), 1 / np.maximum(d, 1), 0)
    else:
        Wm = np.zeros((n, n)); 
        for i in range(n): Wm[i, np.argsort(d[i])[1:7]] = 1
    Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-12); z = v - v.mean()
    I = n / Wm.sum() * (z @ Wm @ z) / (z @ z); perm = []
    for _ in range(nperm): zp = rng.permutation(z); perm.append(n / Wm.sum() * (zp @ Wm @ zp) / (zp @ zp))
    return dict(I=float(I), E=float(-1 / (n - 1)), p=float((np.sum(np.array(perm) >= I) + 1) / (nperm + 1)), n=int(n))
mo = {}
for lab, idx in (("n71", RS.index[~RS.first_filling]), ("n74", RS.index)):
    d = RS.loc[idx]; mo[lab] = {v: {s: moran(d[v].values, d.lon.values, d.lat.values, s) for s in ("invdist300", "knn6")} for v in ("S", "R", "G")}
out["moran"] = mo; print({l: {v: {s: (round(x["I"], 3), x["p"]) for s, x in d_.items()} for v, d_ in m.items()} for l, m in mo.items()})
# (b) nested multilevel (windows in reservoirs in sub-basins), 71 reservoirs
d = WS[~WS.reservoir.isin(FIRST)].dropna(subset=["S", "R", "G"]).copy(); d["basin"] = d.basin.astype(str); ml = {}
for v in ("S", "R", "G"):
    try:
        m = smf.mixedlm(f"{v} ~ 1", d, groups="basin", re_formula="1", vc_formula={"reservoir": "0 + C(reservoir)"}).fit(reml=True)
        tot = float(m.cov_re.iloc[0, 0] + m.vcomp[0] + m.scale)
        ml[v] = dict(mu=float(m.params["Intercept"]), se=float(m.bse["Intercept"]), lo=float(m.params["Intercept"] - 1.96 * m.bse["Intercept"]), hi=float(m.params["Intercept"] + 1.96 * m.bse["Intercept"]),
                     var_basin=float(m.cov_re.iloc[0, 0]), var_res_in_basin=float(m.vcomp[0]), var_win=float(m.scale), share_basin=float(m.cov_re.iloc[0, 0] / tot), share_res=float(m.vcomp[0] / tot), share_win=float(m.scale / tot), n_win=int(len(d)))
    except Exception as e: ml[v] = {"error": str(e)}
    # naive mean over windows with window-level bootstrap for comparison
    ml[v]["naive_window_mean_ci"] = [float(np.percentile(d[v].values[rng.integers(0, len(d), (4000, len(d)))].mean(1), 2.5)), float(np.percentile(d[v].values[rng.integers(0, len(d), (4000, len(d)))].mean(1), 97.5))]
out["multilevel_71"] = ml; print(json.dumps(ml, default=float)[:1500])
# (c) attributes vs R: RF with reservoir-grouped CV, split-half reliability from pixel halves
feat = ["tc_srad_tr0021", "dist_dam_km", "base_ref", "slope", "tc_soil_tr0021", "m_sm", "f_crop", "m_vpd", "f_grass", "f_tree", "f_lcchg", "tc_vpd_tr0021", "d_sm", "age2017", "tc_ppt_tr0021", "loss_late", "m_sw",
        "loss_early", "tc2000", "tc_tmax_tr0021", "m_pr", "d_vpd", "elev", "d_sw", "d_pr", "m_t2m", "f_affor", "d_t2m", "gfc_gain", "f_shrub"]
dd = WS[~WS.reservoir.isin(FIRST)].dropna(subset=["R"]).copy(); Xf = dd[feat].fillna(dd[feat].median()); yv = dd.R.values; g = dd.reservoir.values
def cv_r2(X, y, groups, seeds=5):
    r2 = []
    for s in range(seeds):
        pred = np.zeros(len(y)); ug = np.unique(groups); rr = np.random.default_rng(s); perm = rr.permutation(ug); fold = {u: i % 5 for i, u in enumerate(perm)}; f = np.array([fold[x] for x in groups])
        for k in range(5):
            tr, te = f != k, f == k; m = RandomForestRegressor(300, min_samples_leaf=5, max_features=0.33, random_state=s, n_jobs=-1).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        r2.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(r2)), [float(np.min(r2)), float(np.max(r2))]
out["rf_R"] = dict(zip(["r2", "r2_range"], cv_r2(Xf.values, yv, g)))
# split-half reliability: random halves of upland ring pixels
pix = pd.read_parquet(f"{DC}/pixels.parquet"); pk = pd.concat([pd.read_parquet(f).assign(window=os.path.basename(f)[:-8]) for f in sorted(glob.glob(f"{DC}/peaks/*.parquet"))]) if False else None
import glob
h1, h2 = {}, {}
for w, gp in pix[(pix.cls == "land") & pix.dist.between(60, 300)].groupby("window"):
    if w not in dd.index: continue
    P = pd.read_parquet(f"{DC}/peaks/{w}.parquet"); P = P[P.pid.isin(gp.pid) & P.year.between(2018, 2025)]
    pidu = gp.pid.values; half = rng.random(len(pidu)) < 0.5; res_ = []
    for hm in (half, ~half):
        s = P[P.pid.isin(pidu[hm])].groupby("year").peak.median().reindex(Y18).values
        res_.append(ts_slope_rows(s[None, :], np.asarray(Y18, float))[0] * 1e3)
    h1[w], h2[w] = res_
H = pd.DataFrame({"h1": h1, "h2": h2}).dropna(); r = np.corrcoef(H.h1, H.h2)[0, 1]; out["split_half_R"] = dict(r=float(r), reliability_SB=float(2 * r / (1 + r)), n=int(len(H)))
print(out["rf_R"], out["split_half_R"])
# rank correlations of reservoir-level R with attributes (reservoir means), sub-basin block bootstrap CI
rm = RS.join(W.groupby("reservoir")[feat].mean(), how="left", rsuffix="_w"); sp = {}
for c in ["tc_ppt_tr1321", "tc_tmax_tr1321", "tc_srad_tr1321", "tc_vpd_tr1321", "tc_soil_tr1321", "tc_ppt_tr0021", "tc_tmax_tr0021", "tc_srad_tr0021", "tc_vpd_tr0021", "tc_soil_tr0021", "f_crop", "f_tree", "f_affor", "f_lcchg",
          "loss_late", "gfc_gain", "elev", "slope", "m_t2m", "m_pr", "base_ref", "dist_dam_km", "age2017", "tc2000"]:
    cc = c if c in W.columns else None
    if cc is None: continue
    x = W.groupby("reservoir")[cc].mean().reindex(RS.index[~RS.first_filling]); y = RS.loc[x.index, "R"]; ok = x.notna()
    rho = spearmanr(x[ok], y[ok]); bl = RS.loc[x.index, "basin"].values[ok.values]; ub = np.unique(bl); idx = {b: np.where(bl == b)[0] for b in ub}; xv, yv2 = x[ok].values, y[ok].values; bs = []
    for _ in range(1500):
        pick = rng.choice(ub, len(ub)); i = np.concatenate([idx[b] for b in pick]); bs.append(spearmanr(xv[i], yv2[i])[0])
    sp[c] = dict(rho=float(rho[0]), p=float(rho[1]), ci_block=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=int(ok.sum()))
out["spearman_R_attr_71"] = sp; print({k: (round(v["rho"], 2), round(v["p"], 3)) for k, v in sp.items()})
# (d) sub-basin means and climate trends (TerraClimate Apr-Oct, 2013-2021 and 2000-2021)
sb = RS[~RS.first_filling].groupby("basin").agg(n=("R", "size"), S=("S", "mean"), R=("R", "mean"), G=("G", "mean"), lon=("lon", "mean"), lat=("lat", "mean"))
cl = W[~W.reservoir.isin(FIRST)].groupby("basin")[["tc_ppt_tr1321", "tc_tmax_tr1321", "tc_srad_tr1321", "tc_vpd_tr1321", "tc_soil_tr1321", "tc_ppt_tr0021", "tc_tmax_tr0021", "tc_srad_tr0021", "tc_vpd_tr0021", "tc_soil_tr0021", "f_crop", "f_tree", "f_affor"]].mean()
SB = sb.join(cl); SB.to_csv(f"{OUT}/subbasin_table.csv"); out["subbasin_corr"] = {c: [float(spearmanr(SB[c], SB.R)[0]), float(spearmanr(SB[c], SB.R)[1])] for c in cl.columns}
print(SB[["n", "S", "R", "G"]].round(2).sort_values("R").to_string()); print({k: (round(v[0], 2), round(v[1], 3)) for k, v in out["subbasin_corr"].items()})
# (e) sampling frame: 74 against 441
meta = pd.read_csv(f"{DD}/ywr_meta.csv"); ops = pd.read_csv(f"{DD}/ops_clim.csv")[["GRAND_ID", "type", "minmon", "lon", "lat"]]
frame = meta.merge(ops[["GRAND_ID", "type", "minmon"]], on="GRAND_ID", how="inner"); frame["reach"] = pd.cut(frame.lon, [0, 104, 110, 125], labels=["upper", "middle", "lower"])
dam = W.drop_duplicates("reservoir").set_index("reservoir")[["dam_lon", "dam_lat", "lon", "lat"]]; sampled = set(); match = {}
for r_, rw in dam.iterrows():
    if r_.startswith("YWR_"): gid = int(r_.split("_")[1]); sampled.add(gid); match[r_] = (gid, 0.0); continue
    lo, la = (rw.dam_lon, rw.dam_lat) if np.isfinite(rw.dam_lon) else (rw.lon, rw.lat)
    dd_ = np.hypot((frame.lon - lo) * np.cos(np.radians(la)), frame.lat - la) * 111; j = dd_.idxmin()
    if dd_[j] < 15: sampled.add(int(frame.GRAND_ID[j])); match[r_] = (int(frame.GRAND_ID[j]), float(dd_[j]))
frame["sampled"] = frame.GRAND_ID.isin(sampled)
fr = {"n_frame": int(len(frame)), "n_sampled_matched": int(frame.sampled.sum()), "n_74": 74, "n_52daily_matched": int(sum(1 for k in match if not k.startswith("YWR_")))}
for c in ["type", "reach"]:
    ct = pd.crosstab(frame.sampled, frame[c]); fr[c] = dict(sampled=ct.loc[True].to_dict(), other=ct.loc[False].to_dict(), chi2_p=float(chi2_contingency(ct)[1]))
for c in ["CAP_MAX", "n_dz", "swo_res_m"]:
    a, b = frame.loc[frame.sampled, c].dropna(), frame.loc[~frame.sampled, c].dropna()
    fr[c] = dict(sampled_median=float(a.median()), other_median=float(b.median()), sampled_iqr=[float(a.quantile(.25)), float(a.quantile(.75))], other_iqr=[float(b.quantile(.25)), float(b.quantile(.75))], mw_p=float(mannwhitneyu(a, b).pvalue), n_a=int(len(a)), n_b=int(len(b)))
frame.to_csv(f"{OUT}/frame441.csv", index=False); out["frame"] = fr; print(json.dumps(fr, default=float)[:2500])
# human footprint proxies for sampled windows: windows per reservoir
dump(out, "r5_spatial")
