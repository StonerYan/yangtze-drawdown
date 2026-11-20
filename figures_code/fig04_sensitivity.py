"""Fig. 4 - Clear-sky observations per year and sensitivity of S, R and G (71 reservoirs) to window, composite, resampling and upland definition."""
from figkit import *

J = json.load(open(f"{AN}/r1_core.json", encoding="utf-8")); B = J["B"]; C = J["C"]
fig = plt.figure(figsize=(W + 3 * MM, 124 * MM))
gs = fig.add_gridspec(2, 5, width_ratios=[0.88, 0.62, 0.55, 0.55, 0.55], height_ratios=[1, 1], hspace=0.62, wspace=0.14, left=0.069, right=0.985, top=0.94, bottom=0.095)
yrs = list(range(2017, 2026))

# ---- a: clear looks ----
a = fig.add_subplot(gs[0, 0])
ring = [C["nobs_ring_by_year"][str(y)] for y in yrs]; shore = [C["nobs_shore_by_year"][str(y)] for y in yrs]
cr = [P["salmon"] if y == 2017 else P["teal"] for y in yrs]; cs = [P["amber"] if y == 2017 else P["blue"] for y in yrs]
xx = np.arange(len(yrs)); a.bar(xx - 0.2, ring, 0.4, color=cr); a.bar(xx + 0.2, shore, 0.4, color=cs)
a.set_xticks(xx); a.set_xticklabels([str(y)[2:] for y in yrs], fontsize=FS["tick"]); a.set_xlabel("Year (20xx)", labelpad=1)
a.set_ylabel("Cloud-free obs. per window"); a.set_ylim(0, 24); a.set_xlim(-0.85, 8.6)
a.annotate("2017: 6.6", xy=(-0.2, 6.63), xytext=(-0.65, 19.5), fontsize=FS["small"], color=DK["salmon"], fontweight="bold", ha="left", va="center", arrowprops=dict(arrowstyle="-", color=DK["salmon"], lw=0.8, relpos=(0.15, 0.0)))
ttl(a, "Cloud-free observations"); panel(a, "a", x=-0.22, y=1.09)

# ---- b: valid-pixel fraction ----
b = fig.add_subplot(gs[1, 0]); vp = C["valid_pixel_fraction"]
b.plot(yrs, [vp["valid_ring"][str(y)] for y in yrs], "-o", color=DK["teal"], ms=4, lw=1.8, label="Upland")
b.plot(yrs, [vp["valid_dz"][str(y)] for y in yrs], "-o", color=C_S, ms=4, lw=1.8, label="Drawdown zone")
b.set_ylim(0, 1.05); b.set_xticks([2017, 2019, 2021, 2023, 2025]); b.set_ylabel("Share of pixels with a value")
b.legend(loc="lower right", fontsize=FS["small"], handlelength=1.4)
ttl(b, "Valid pixels"); panel(b, "b", x=-0.22, y=1.09)

# ---- c,d,e: forest plots ----
rows = []   # (label, group, R(m,lo,hi), S, G)
def trip(d): return (d["mean"], d["lo"], d["hi"])
W71 = {k: v["n71"] for k, v in B["windows"].items()}
rows.append(("Primary, 2018–2025", "Trend window", trip(W71["2018-2025"]["R"]), trip(W71["2018-2025"]["S"]), trip(W71["2018-2025"]["G"])))
for lab, key in [("2017–2025", "2017-2025 (original)"), ("2019–2025", "2019-2025"), ("2018–2024", "2018-2024"), ("2018–2023", "2018-2023")]:
    rows.append((lab, "", trip(W71[key]["R"]), trip(W71[key]["S"]), trip(W71[key]["G"])))
CM = {k: v["n71"] for k, v in C["composites"].items()}
for lab, key in [("Fixed six observations", "fixed six looks"), ("July–September mean", "Jul-Sep mean")]:
    rows.append((lab, "Composite" if lab.startswith("Fixed") else "", trip(CM[key]["R"]), trip(CM[key]["S"]), trip(CM[key]["G"])))
ar = C["R_trend_density_adj_71"]["adj"]; as_ = C["S_trend_density_adj_71"]["adj"]
rows.append(("Obs.-count adjusted", "", tuple(ar), tuple(as_), None))
yb = B["year_block"]["n71"]
pr = lambda k: [W71["2018-2025"][k]["mean"]] + yb[k]["year_ci"]
rows.append(("Resampling years", "Interval", tuple(pr("R")), tuple(pr("S")), tuple(pr("G"))))
for lab, key in [("300–600 m", "R_300_600"), ("600–1,050 m", "R_600_1050"), ("Slope-matched", "R_slope"), ("Stable cover", "R_stable"), ("Natural cover", "R_natural"), ("All upland", "R_regional")]:
    ci = B["refdefs_ci"][key]["n71"]; g = B["refdefs"][key]["n71"]["G_" + key[2:] if False else "G_" + key]
    rows.append((lab, "Upland control" if lab == "300–600 m" else "", (ci["mean"], ci["lo"], ci["hi"]), None, (g, None, None)))
n = len(rows); ypos = np.arange(n)[::-1]
axs = []
for j, (v, colr, dk_, ttl_, lett) in enumerate([("R", C_R, DK["teal"], "Upland (R)", "c"), ("S", C_S, DK["blue"], "Drawdown (S)", "d"), ("G", C_G, DK["red"], "Difference (G)", "e")]):
    ax = fig.add_subplot(gs[:, 2 + j]) if j == 0 else fig.add_subplot(gs[:, 2 + j], sharey=axs[0]); axs.append(ax)
    for i, r in enumerate(rows):
        t_ = r[2 + j]
        if t_ is None: continue
        m, lo, hi = t_; y = ypos[i]
        if lo is None: ax.plot(m, y, "D", color=dk_, ms=4.2, mfc="white", mew=1.1)
        else:
            ax.plot([lo, hi], [y, y], color=colr, lw=2.4, solid_capstyle="round"); ax.plot(m, y, "o", color=dk_, ms=4.6)
    ax.axvline(0, color="#777", lw=0.8); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(-0.8, n - 0.2); ax.spines["left"].set_visible(j == 0)
    if j > 0: ax.tick_params(axis="y", left=False, labelleft=False)
    ttl(ax, ttl_); ax.set_xlabel("EVI2 × 10$^{-3}$ yr$^{-1}$", fontsize=FS["small"], labelpad=1)
    ax.title.set_position((0.0 if j else -0.0, 1.0))
    panel(ax, lett, x=-0.04, y=1.045)
axs[0].set_yticks(ypos); axs[0].set_yticklabels([])
lb = fig.add_subplot(gs[:, 1], sharey=axs[0]); lb.axis("off")
for i, r in enumerate(rows):
    lb.text(1.0, ypos[i], r[0], ha="right", va="center", fontsize=FS["tick"], transform=lb.get_yaxis_transform())
    if r[1] and i > 0:
        for ax_ in axs: ax_.axhline(ypos[i] + 0.5, color="#BBBBBB", lw=0.8, ls=(0, (3, 2)))
axs[0].set_xlim(-2, 10.5); axs[1].set_xlim(-5.5, 3.5); axs[2].set_xlim(-16, 1)
align_panels(fig, rows=[[a, axs[0], axs[1], axs[2]]], cols=[[a, b]])
save(fig, "Fig04_sensitivity")
