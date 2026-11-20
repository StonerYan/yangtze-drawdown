"""Fig. 5 - Upland control in Landsat (2000-2025), MODIS and Sentinel-2, and the pre-impoundment contrast."""
from figkit import *
from scipy import stats

J = json.load(open(f"{AN}/r2_longrecord.json", encoding="utf-8")); X = pd.read_csv(f"{AN}/baci_landsat.csv"); PR = pd.read_csv(f"{AN}/s2_landsat_pairs18.csv")
L = pd.read_parquet(f"{D2}/landsat_density_winyear.parquet"); L = L[L.set == "harm"].dropna(subset=["R_fix3"])
MS = pd.read_parquet(f"{D2}/modis_res_series.parquet"); MS = MS[MS.system == "reservoir"]
rng = np.random.default_rng(11)
fig = plt.figure(figsize=(W + 3 * MM, 124 * MM))
gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.22, 1, 1], height_ratios=[1, 1], hspace=0.62, wspace=1.10, left=0.069, right=0.985, top=0.94, bottom=0.10)

# ---- a: Landsat series with fitted models ----
a = fig.add_subplot(gs[0, :4])
L = L.assign(an=L.R_fix3 - L.groupby("window").R_fix3.transform("mean"))
H = L.groupby(["reservoir", "year"]).an.mean().unstack() * 1e3
yrs = H.columns.values; m = H.mean(0).values
bs = np.array([np.nanmean(H.values[rng.integers(0, len(H), len(H))], 0) for _ in range(2000)])
a.axvspan(1999.5, 2012.5, color="#F1F1F1", lw=0); a.axvspan(2012.5, 2025.5, color="#E4E4E4", lw=0)
a.text(2006, 0.97, "Landsat 5 and 7", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=FS["small"], color="#555")
a.text(2019, 0.97, "Landsat 8 and 9", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=FS["small"], color="#555")
a.fill_between(yrs, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), color=C_R, alpha=0.30, lw=0); a.plot(yrs, m, "-o", color=DK["teal"], ms=3.6, lw=1.6)
Mh = J["models"]["harm:R_fix3:2000-2025"]; t = yrs - 2000.
def fitline(kind):
    if kind == "step": g = lambda tt: Mh["step"]["slope"] * (tt - 2000) + Mh["step"]["step"] * (tt + 2000 >= 2013)
    else:
        tau, pre, post = Mh["hinge"]["tau"], Mh["hinge"]["pre"], Mh["hinge"]["post"]; g = lambda tt: pre * (tt) + (post - pre) * np.maximum(0, tt + 2000 - tau)
    yy = g(t); return yy - yy.mean() + m.mean()
xx = np.linspace(2000, 2025, 300)
a.plot(yrs, fitline("hinge"), color=INK, lw=2.0, label=f"Segmented, break {Mh['hinge']['tau']:.0f}")
a.plot(yrs, fitline("step"), color=DK["red"], lw=2.0, ls=(0, (4, 2)), label="Sensor step at 2013")
a.axhline(0, color="#777", lw=0.8); a.set_xlim(1999.5, 2025.5); a.set_xticks(range(2000, 2026, 5)); a.set_ylabel("Upland anomaly\n(EVI2 × 10$^{-3}$)")
a.legend(loc="lower right", fontsize=FS["small"], handlelength=2.2, ncol=2, columnspacing=1.2)
ttl(a, "Landsat, 34 reservoirs, harmonised"); panel(a, "a", x=-0.07, y=1.08)

# ---- b: slopes by sensor family / period ----
b = fig.add_subplot(gs[0, 5:])
ss = J["single_sensor"]; ex = J["exclude_2011_2014"]["harm:R_fix3"]
rows = [("Landsat 7\n2000–2012", ss["ETM:2000-2012"]["mean"], ss["ETM:2000-2012"]["lo"], ss["ETM:2000-2012"]["hi"]),
        ("Landsat 7\n2013–2019", ss["ETM:2013-2019"]["mean"], ss["ETM:2013-2019"]["lo"], ss["ETM:2013-2019"]["hi"]),
        ("Landsat 8\n2013–2025", ss["OLI:2013-2025"]["mean"], ss["OLI:2013-2025"]["lo"], ss["OLI:2013-2025"]["hi"]),
        ("Landsat 8\n2018–2025", ss["OLI:2018-2025"]["mean"], ss["OLI:2018-2025"]["lo"], ss["OLI:2018-2025"]["hi"]),
        ("Before 2011", *ex["pre_2000_2010"]), ("After 2014", *ex["post_2015_2025"])]
n = len(rows); yp = np.arange(n)[::-1]
for i, (lab, mm, lo, hi) in enumerate(rows):
    col = C_R if i < 4 else P["amber"]; dk_ = DK["teal"] if i < 4 else DK["amber"]
    b.plot([lo, hi], [yp[i], yp[i]], color=col, lw=2.6, solid_capstyle="round"); b.plot(mm, yp[i], "o", color=dk_, ms=4.8)
b.axvline(0, color="#777", lw=0.8); b.set_yticks(yp); b.set_yticklabels([r[0] for r in rows], fontsize=FS["small"]); b.grid(axis="x", color=GRID, lw=0.6); b.set_axisbelow(True)
b.axhline(1.5 + 0.0, color="#BBBBBB", lw=0.8, ls=(0, (3, 2)))
b.set_xlabel("Upland trend (EVI2 × 10$^{-3}$ yr$^{-1}$)", fontsize=FS["small"], labelpad=1); b.set_ylim(-0.6, n - 0.4)
ttl(b, "Single-sensor trends"); panel(b, "b", x=-0.46, y=1.08)

# ---- c: MODIS ----
c = fig.add_subplot(gs[1, :2])
ms = MS.copy(); ms["a"] = ms.evi - ms.groupby("reservoir").evi.transform("mean"); mm_ = ms.groupby("year").a.agg(["mean", "sem"])
c.fill_between(mm_.index, (mm_["mean"] - 1.96 * mm_["sem"]) * 1e3, (mm_["mean"] + 1.96 * mm_["sem"]) * 1e3, color=P["amber"], alpha=0.40, lw=0); c.plot(mm_.index, mm_["mean"] * 1e3, "-o", color=DK["amber"], ms=3.4, lw=1.6)
c.axhline(0, color="#777", lw=0.8); c.set_xticks(range(2005, 2025, 5)); c.set_ylabel("EVI anomaly (× 10$^{-3}$)")
mo = J["modis"]
for (x0, x1, key) in [(2001, 2012, "evi:2001-2012"), (2013, 2019, "evi:2013-2019"), (2018, 2024, "evi:2018-2025")]:
    c.plot([x0, x1], [-31 if key != "evi:2018-2025" else -39] * 2, color=INK, lw=2.0, solid_capstyle="butt")
    c.text((x0 + x1) / 2, -31.5 if key != "evi:2018-2025" else -39.5, f"{mo[key]['mean']:+.1f}".replace("-", "−"), ha="center", va="top", fontsize=FS["small"], fontweight="bold")
c.set_ylim(-52, 36)
ttl(c, "MODIS, 74 reservoirs"); panel(c, "c", x=-0.30, y=1.08)

# ---- d: Sentinel-2 vs Landsat ----
d = fig.add_subplot(gs[1, 2:4])
x_, y_ = PR.landsat, PR.s2; lim = [min(x_.min(), y_.min()) - 2, max(x_.max(), y_.max()) + 2]
d.plot(lim, lim, color="#999", lw=1.0, ls="--"); d.scatter(x_, y_, s=22, color=C_R, edgecolor=DK["teal"], lw=0.6, zorder=3)
r = J["s2_vs_landsat_2018_2025"]
d.text(0.05, 0.96, f"r = {r['r']:.2f}\nn = {r['n']}", transform=d.transAxes, fontsize=FS["ann"], va="top", fontweight="bold")
d.set_xlim(lim); d.set_ylim(lim); d.set_aspect("equal"); d.set_xlabel("Landsat 8 trend", labelpad=1); d.set_ylabel("Sentinel-2 trend", labelpad=1)
ttl(d, "Two sensors, 2018–2025"); panel(d, "d", x=-0.36, y=1.08)

# ---- e: pre-impoundment contrast ----
e = fig.add_subplot(gs[1, 5:])
bc = J["baci"]; items = [("S", bc["pre_S_fix3"], C_S), ("R", bc["pre_R_fix3"], C_R), ("S−R", bc["pre_diff"], C_G), ("S", bc["post_S_fix3"], C_S), ("R", bc["post_R_fix3"], C_R), ("S−R", bc["post_diff"], C_G)]
xs = [0, 1, 2, 3.6, 4.6, 5.6]
for x0, (lab, d_, col) in zip(xs, items):
    e.bar(x0, d_["mean"], color=col, width=0.7); e.errorbar(x0, d_["mean"], yerr=[[d_["mean"] - d_["lo"]], [d_["hi"] - d_["mean"]]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5)
e.axhline(0, color="#777", lw=0.8); e.set_xticks(xs); e.set_xticklabels([i[0] for i in items], fontsize=FS["small"])
e.text(1, 0.99, "Before", transform=e.get_xaxis_transform(), ha="center", va="top", fontsize=FS["small"], fontweight="bold", color="#555")
e.text(4.6, 0.99, "After", transform=e.get_xaxis_transform(), ha="center", va="top", fontsize=FS["small"], fontweight="bold", color="#555")
e.set_ylabel("Trend (EVI2 × 10$^{-3}$ yr$^{-1}$)", fontsize=FS["small"]); e.set_ylim(-11, 7)
ttl(e, "Impoundment, 18 reservoirs"); panel(e, "e", x=-0.42, y=1.08)
align_panels(fig, rows=[[a, b], [c, d, e]], cols=[[a, c], [b, e]])
save(fig, "Fig05_longrecord")
