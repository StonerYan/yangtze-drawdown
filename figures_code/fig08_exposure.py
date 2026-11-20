"""Fig. 8 - Time-varying exposure from daily storage: validation, within-pixel association with peak EVI2, and accounting for trends."""
from figkit import *

R3 = json.load(open(f"{AN}/r3_exposure.json", encoding="utf-8")); Rg = R3["reg"]
V = pd.read_csv(f"{AN}/exposure_validation_all.csv"); AC = pd.read_csv(f"{AN}/exposure_accounting.csv")
fig = plt.figure(figsize=(W, 112 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1], hspace=0.60, wspace=0.42, left=0.075, right=0.985, top=0.94, bottom=0.10)

# ---- a: validation ----
a = fig.add_subplot(gs[0, 0])
a.hist(V.bal_acc, bins=np.linspace(0.4, 1.0, 25), color=C_S, edgecolor="white", lw=0.7)
a.axvline(V.bal_acc.median(), color=C_G, lw=2.0); a.axvline(0.7, color="#777", lw=1.0, ls="--")
yl = a.get_ylim()[1]; a.text(V.bal_acc.median() + 0.012, yl * 0.97, f"median {V.bal_acc.median():.2f}", color=DK["red"], fontsize=FS["small"], fontweight="bold", va="top")
a.set_xlabel("Balanced accuracy of modelled exposure", labelpad=1); a.set_ylabel("Windows (n = 464)")
ttl(a, "Validation"); panel(a, "a", x=-0.22, y=1.08)

# ---- b: coefficients ----
b = fig.add_subplot(gs[:, 1])
rows = [("All pixels", "A", Rg["A_pixel_year"]["exp_gs"]), ("All pixels", "B", Rg["B_pixel_resyear"]["exp_gs"]),
        ("Early season", "A", Rg["A_early_late"]["exp_early"]), ("Early season", "B", Rg["B_early_late"]["exp_early"]),
        ("Late season", "A", Rg["A_early_late"]["exp_late"]), ("Late season", "B", Rg["B_early_late"]["exp_late"])]
for k, lab in [("A_F15_35", "15–35% band"), ("A_F35_55", "35–55% band"), ("A_F55_75", "55–75% band"), ("A_F75_95", "75–95% band")]:
    rows.append((lab, "A", Rg[k]["exp_gs"]))
n = len(rows); yp = np.arange(n)[::-1].astype(float); yp[2:] -= 0.35; yp[6:] -= 0.35
for i, (lab, spec, e) in enumerate(rows):
    col, dk_ = (C_S, DK["blue"]) if spec == "A" else (P["amber"], DK["amber"])
    b.plot([e["lo"], e["hi"]], [yp[i]] * 2, color=col, lw=2.6, solid_capstyle="round"); b.plot(e["beta"], yp[i], "o", color=dk_, ms=4.8)
b.axvline(0, color="#777", lw=0.8); b.grid(axis="x", color=GRID, lw=0.6); b.set_axisbelow(True)
b.set_yticks(yp); b.set_yticklabels([r[0] for r in rows], fontsize=FS["tick"])
for yy in [yp[1] - 0.5 - 0.17, yp[3] - 0.5 - 0.17, yp[5] - 0.5 - 0.17]: b.axhline(yy, color="#BBBBBB", lw=0.8, ls=(0, (3, 2)))
b.plot([], [], "o-", color=DK["blue"], label="Pixel and year effects (A)"); b.plot([], [], "o-", color=DK["amber"], label="Pixel and reservoir-by-year effects (B)")
b.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=FS["small"], handlelength=1.6, labelspacing=0.3)
b.set_xlabel("Change in peak EVI2 per unit exposure fraction", labelpad=1); b.set_xlim(-0.06, 0.21); b.set_ylim(yp[-1] - 0.6, yp[0] + 1.85)
ttl(b, "Association with exposure"); panel(b, "b", x=-0.50, y=1.035)

# ---- c: accounting ----
c = fig.add_subplot(gs[1, 0])
c.scatter(AC.pred, AC.obs, s=18, color=C_S, edgecolor=DK["blue"], lw=0.5, zorder=3)
lim = [-10, 16]; c.plot([-2.6, 2.6], [-2.6, 2.6], color="#999", lw=1.0, ls="--")
c.axhline(0, color="#777", lw=0.8); c.axvline(0, color="#777", lw=0.8)
c.text(0.04, 0.96, f"r = {R3['accounting']['r']:.2f}\nn = {R3['accounting']['n_res']}", transform=c.transAxes, fontsize=FS["ann"], va="top", fontweight="bold")
c.set_xlabel("Trend predicted from exposure", labelpad=1); c.set_ylabel("Observed pixel-mean trend", labelpad=1)
ttl(c, "Reservoir trends"); panel(c, "c", x=-0.22, y=1.08)
align_panels(fig, rows=[[a, b]], cols=[[a, c]])
save(fig, "Fig08_exposure")
