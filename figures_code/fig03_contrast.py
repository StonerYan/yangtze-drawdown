"""Fig. 3 - S, R and G at the 71 reservoirs in operation throughout 2018-2025, and the counterfactual bracket."""
from figkit import *
from scipy.stats import gaussian_kde

J = json.load(open(f"{AN}/r1_core.json", encoding="utf-8")); M6 = json.load(open(f"{AN}/r6_misc.json", encoding="utf-8"))
RS = pd.read_csv(f"{AN}/res18.csv"); FIRSTS = RS[RS.first_filling].reservoir.tolist(); R71 = RS[~RS.first_filling]
WY = pd.read_parquet(f"{ROOT}/data/cee/winyear.parquet"); WY = WY[WY.year.between(2018, 2025)]
WT = pd.read_parquet(f"{D2}/win_table.parquet").drop_duplicates("window").set_index("window").reservoir
WY["reservoir"] = WY.window.map(WT)
ann = WY[~WY.reservoir.isin(FIRSTS)].groupby(["reservoir", "year"])[["shore", "R_60_300", "G_R_60_300"]].mean() * 1e3
rng = np.random.default_rng(5)
fig = plt.figure(figsize=(W, 118 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1, 1], hspace=0.50, wspace=0.34, left=0.075, right=0.985, top=0.945, bottom=0.095)
cols = {"S": C_S, "R": C_R, "G": C_G}; dk = {"S": DK["blue"], "R": DK["teal"], "G": DK["red"]}
nm = {"S": "Drawdown zone (S)", "R": "Upland control (R)", "G": "Difference (G)"}

# ---- a: annual anomalies ----
a = fig.add_subplot(gs[0, 0])
for v, c in [("G", "G_R_60_300"), ("R", "R_60_300"), ("S", "shore")]:
    P_ = ann[c].unstack(); P_ = P_.dropna(); A_ = P_.sub(P_.mean(1), axis=0); n_ = len(A_)
    bs = np.array([A_.values[rng.integers(0, n_, n_)].mean(0) for _ in range(2000)]); m = A_.mean(0).values; yrs = A_.columns.values
    a.fill_between(yrs, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), color=cols[v], alpha=0.30, lw=0)
    a.plot(yrs, m, "-o", color=cols[v], lw=2.0, ms=4.2, zorder=4)
    b_ = np.polyfit(yrs, m, 1); a.plot(yrs, np.polyval(b_, yrs), "--", color=cols[v], lw=1.0)
    a.text(2025.25, m[-1], nm[v], color=dk[v], fontsize=FS["ann"], fontweight="bold", va="center")
a.plot([2017.6, 2025.0], [0, 0], color="#777777", lw=0.8, zorder=1)
a.set_ylabel("Anomaly (EVI2 × 10$^{-3}$)"); a.set_xticks(range(2018, 2026)); a.set_xlim(2017.6, 2028.2)
a.spines["bottom"].set_bounds(2017.6, 2025.4)
ttl(a, "Annual anomalies, 71 reservoirs"); panel(a, "a", x=-0.115, y=1.08)

# ---- b: trends with sub-basin CIs ----
b = fig.add_subplot(gs[0, 1])
A71 = J["A"]["n71"]; xs = [0, 1, 2]; ks = ["S", "R", "G"]
for x_, k in zip(xs, ks):
    m = A71[k]["mean"]; lo, hi = A71[k]["sub_basin_ci"]
    b.bar(x_, m, color=cols[k], width=0.62); b.errorbar(x_, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, capthick=1.0)
    b.text(x_, hi + 0.5 if m > 0 else lo - 0.5, f"{m:+.1f}".replace("-", "−"), ha="center", va="bottom" if m > 0 else "top", fontsize=FS["ann"] + 0.5, fontweight="bold")
b.axhline(0, color="#777", lw=0.8); b.set_xticks(xs); b.set_xticklabels(ks, fontsize=FS["label"], fontweight="bold")
b.set_ylabel("Trend (EVI2 × 10$^{-3}$ yr$^{-1}$)"); b.set_ylim(-11.5, 9.0); b.set_xlim(-0.55, 2.55)
ttl(b, "Trends, 2018–2025"); panel(b, "b", x=-0.30, y=1.08)

# ---- c: reservoir distributions ----
c = fig.add_subplot(gs[1, 0])
ys = {"G": 2, "S": 1, "R": 0}; vals = {"G": R71.G, "S": R71.S, "R": R71.R}; xx = np.linspace(-45, 45, 400)
for k in ["G", "S", "R"]:
    v = vals[k].values; kde = gaussian_kde(v, bw_method=0.35)(xx); kde = kde / kde.max() * 0.45
    c.fill_between(xx, ys[k], ys[k] + kde, color=cols[k], alpha=0.45, lw=0); c.plot(xx, ys[k] + kde, color=cols[k], lw=1.2)
    jit = np.random.default_rng(3).uniform(-0.24, -0.05, len(v)); c.scatter(v, ys[k] + jit, s=6, color=cols[k], alpha=0.75, lw=0)
    c.plot([np.median(v)] * 2, [ys[k] - 0.27, ys[k] + 0.5], color=INK, lw=1.4)
    c.text(49, ys[k] + 0.12, f"{(v < 0).sum()} of 71", fontsize=FS["ann"], va="center", ha="right", fontweight="bold")
c.axvline(0, color="#777", lw=0.8); c.set_xlim(-45, 52); c.set_ylim(-0.4, 2.85)
c.set_yticks([0, 1, 2]); c.set_yticklabels(["R", "S", "G"], fontsize=FS["label"], fontweight="bold")
for t_, k in zip(c.get_yticklabels(), ["R", "S", "G"]): t_.set_color(dk[k])
c.text(49, 2.80, "reservoirs < 0", fontsize=FS["small"], ha="right", va="top", color="#555")
c.set_xlabel("Reservoir trend (EVI2 × 10$^{-3}$ yr$^{-1}$)"); c.spines["left"].set_visible(False); c.tick_params(axis="y", length=0)
ttl(c, "Reservoir trends"); panel(c, "c", x=-0.115, y=1.04)

# ---- d: counterfactual bracket ----
d = fig.add_subplot(gs[1, 1])
SC = M6["scenarios"]["n71"]; lam = np.array([float(k) for k in SC]); mm = np.array([float(SC[k][0]) for k in SC])
lo = np.array([float(SC[k][1]) for k in SC]); hi = np.array([float(SC[k][2]) for k in SC])
d.fill_between(lam, lo, hi, color=C_G, alpha=0.22, lw=0); d.plot(lam, mm, "-o", color=C_G, lw=2.0, ms=4.5)
d.axhline(0, color="#777", lw=0.8)
RA = J["D"]["regression_adjustment"]["n71"]
d.errorbar([0.0], [RA["delta"]], yerr=[[RA["delta"] - RA["delta_ci"][0]], [RA["delta_ci"][1] - RA["delta"]]], fmt="s", color=INK, ms=5, capsize=3, lw=1.1, zorder=6, clip_on=False)
d.annotate("regression-\nadjusted", xy=(0.0, RA["delta"]), xytext=(0.30, 1.0), fontsize=FS["small"], color=INK, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
d.set_xlim(-0.06, 1.06); d.set_ylim(-9, 2.2); d.set_xlabel("λ, share of the upland trend\nthat S would have followed"); d.set_ylabel("Effect on trend\n(EVI2 × 10$^{-3}$ yr$^{-1}$)")
d.set_xticks([0, 0.5, 1]); d.set_xticklabels(["0\nS stable", "0.5", "1\nS follows R"], fontsize=FS["tick"])
d.set_xlabel("λ, share of the upland trend S would have followed", labelpad=1, fontsize=FS["small"] + 0.5)
ttl(d, "Counterfactual bounds"); panel(d, "d", x=-0.40, y=1.04)
align_panels(fig, rows=[[a, b], [c, d]], cols=[[a, c], [b, d]])
save(fig, "Fig03_contrast")
