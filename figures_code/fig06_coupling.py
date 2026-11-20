"""Fig. 6 - Year-to-year coupling between drawdown zone (S) and upland control (R), 70 reservoirs with complete series."""
from figkit import *

J = json.load(open(f"{AN}/r1_core.json", encoding="utf-8")); D71 = J["D"]["n71"] if "n71" in J["D"] else None
DT = pd.read_csv(f"{AN}/detrended18.csv"); RS = pd.read_csv(f"{AN}/res18.csv"); FIRSTS = RS[RS.first_filling].reservoir.tolist()
DT = DT[~DT.reservoir.isin(FIRSTS)]; nres = DT.reservoir.nunique()
rng = np.random.default_rng(8)
# per-reservoir correlations
cr = DT.groupby("reservoir").apply(lambda g: np.corrcoef(g.eS, g.eR)[0, 1]).values
Dk = [k for k in J["D"].keys()]
Dd = J["D"]
nul = None
for k in Dd:
    if isinstance(Dd[k], dict) and "null_median_ci" in Dd[k] and Dd[k].get("n") == nres: nul = Dd[k]
if nul is None:
    for k in Dd:
        if isinstance(Dd[k], dict) and "null_median_ci" in Dd[k]: print("coupling key", k, Dd[k]["n"])
coup = nul or [Dd[k] for k in Dd if isinstance(Dd[k], dict) and "null_median_ci" in Dd[k]][-1]
RA = Dd["regression_adjustment"]["n71"]
# 2022 anomalies (70 reservoirs with complete series)
WY = pd.read_parquet(f"{ROOT}/data/cee/winyear.parquet"); WY = WY[WY.year.between(2018, 2025)]
WT = pd.read_parquet(f"{D2}/win_table.parquet").drop_duplicates("window").set_index("window").reservoir; WY["reservoir"] = WY.window.map(WT)
ann = WY[~WY.reservoir.isin(FIRSTS)].groupby(["reservoir", "year"])[["shore", "R_60_300", "G_R_60_300"]].mean() * 1e3
an22 = {}
for v, c in [("S", "shore"), ("R", "R_60_300"), ("G", "G_R_60_300")]:
    P_ = ann[c].unstack().dropna(); A_ = P_.sub(P_.mean(1), axis=0)[2022].values
    bs = A_[rng.integers(0, len(A_), (4000, len(A_)))].mean(1); an22[v] = (A_.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(A_))
print("n res", nres, "an22", an22)
fig = plt.figure(figsize=(W, 112 * MM))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.36, left=0.08, right=0.985, top=0.94, bottom=0.10)

# ---- a: residual scatter ----
a = fig.add_subplot(gs[0, 0])
a.scatter(DT.eR, DT.eS, s=7, color=C_S, alpha=0.40, lw=0)
lim = 100; xx = np.array([-lim, lim])
a.plot(xx, xx, color="#999", ls="--", lw=1.0); bt = np.polyfit(DT.eR, DT.eS, 1)
a.plot(xx, np.polyval(bt, xx), color=C_G, lw=2.0)
a.text(0.04, 0.96, "dashed: slope 1\nred: pooled slope %.2f" % bt[0], transform=a.transAxes, fontsize=FS["small"], va="top", color="#444")
a.set_xlim(-130, 130); a.set_ylim(-lim, lim); a.set_aspect("equal")
a.set_xlabel("R residual (EVI2 × 10$^{-3}$)", labelpad=1); a.set_ylabel("S residual (EVI2 × 10$^{-3}$)", labelpad=1)
ttl(a, "Detrended annual values"); panel(a, "a", x=-0.19, y=1.06)

# ---- b: per-reservoir correlations ----
b = fig.add_subplot(gs[0, 1])
lo, hi = coup["null_median_ci"]
b.axvspan(lo, hi, color="#DDDDDD", lw=0, zorder=0)
b.hist(cr, bins=np.linspace(-1, 1, 21), color=C_S, edgecolor="white", lw=0.8)
b.axvline(np.median(cr), color=C_G, lw=2.0); b.axvline(0, color="#777", lw=0.8)
yl = b.get_ylim()[1]; b.set_ylim(0, yl * 1.28); yl_top = yl * 1.25
b.text(np.median(cr) + 0.04, yl_top, f"median {np.median(cr):.2f}", color=DK["red"], fontsize=FS["small"], fontweight="bold", va="top")
b.text(-0.98, yl_top, "grey band: range of\nthe median under\nindependence", color="#555", fontsize=FS["small"], va="top", ha="left", linespacing=1.15)
b.set_xlabel("Correlation of S and R residuals", labelpad=1); b.set_ylabel("Reservoirs")
ttl(b, "Per-reservoir correlation"); panel(b, "b", x=-0.25, y=1.06)

# ---- c: regression coefficient and variability ----
c = fig.add_subplot(gs[1, 0])
sd = [coup["sdS"], coup["sdR"], coup["sdG"]]; cols = [C_S, C_R, C_G]
c.bar([0, 1, 2], sd, color=cols, width=0.62)
exp_g = coup["ratio_expected_if_independent"] * coup["sdS"]
c.plot([1.6, 2.4], [exp_g] * 2, color=INK, lw=1.6, ls=(0, (3, 2)))
c.text(2.45, exp_g, "expected if\nindependent", fontsize=FS["small"], va="center", ha="left")
for x_, v in zip([0, 1, 2], sd): c.text(x_, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=FS["ann"], fontweight="bold")
c.set_xticks([0, 1, 2]); c.set_xticklabels(["S", "R", "G"], fontweight="bold", fontsize=FS["label"]); c.set_xlim(-0.6, 3.5); c.set_ylim(0, 44)
c.set_ylabel("SD of residuals\n(EVI2 × 10$^{-3}$)")
c.text(0.02, 0.97, f"coefficient on R: {RA['beta']:.2f}\n({RA['beta_ci'][0]:.2f} to {RA['beta_ci'][1]:+.2f})".replace("-", "−"), transform=c.transAxes, fontsize=FS["small"], va="top", fontweight="bold", color=INK)
ttl(c, "Variability and adjustment"); panel(c, "c", x=-0.25, y=1.06)

# ---- d: 2022 ----
d = fig.add_subplot(gs[1, 1])
for x_, (k, col) in enumerate([("S", C_S), ("R", C_R), ("G", C_G)]):
    m, l, h, n = an22[k]; d.bar(x_, m, color=col, width=0.62); d.errorbar(x_, m, yerr=[[m - l], [h - m]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=3)
d.axhline(0, color="#777", lw=0.8); d.set_xticks([0, 1, 2]); d.set_xticklabels(["S", "R", "G"], fontweight="bold", fontsize=FS["label"])
d.set_ylabel("2022 anomaly\n(EVI2 × 10$^{-3}$)"); d.set_ylim(-26, 14)
ttl(d, "Growing season of 2022"); panel(d, "d", x=-0.25, y=1.06)
align_panels(fig, rows=[[a, b], [c, d]], cols=[[a, c], [b, d]])
save(fig, "Fig06_coupling")
json.dump({k: [float(x) for x in v] for k, v in an22.items()}, open(f"{AN}/anom2022_n71.json", "w"))
print("cr median", np.median(cr), "n pos", (cr > 0).sum(), len(cr), "coup keys", {k: coup[k] for k in ("sdS", "sdR", "sdG", "ratio", "null_median_ci")})
