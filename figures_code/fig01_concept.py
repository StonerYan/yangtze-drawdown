"""Fig. 1 - Concept: the contrast G = S - R, the two counterfactuals that bracket the effect, and the design of the evidence."""
from figkit import *
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpecFromSubplotSpec

fig = plt.figure(figsize=(W, 140 * MM))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.56], hspace=0.14, left=0.012, right=0.992, top=0.985, bottom=0.015)
top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[1.08, 1.0], wspace=0.10)
SOIL, SOIL_E = "#EBE3D5", "#9A8A70"

# ======================= a: valley cross-section =======================
a = fig.add_subplot(top[0]); a.set_xlim(0, 100); a.set_ylim(-6, 78); a.axis("off")
x = np.linspace(0, 100, 600)
prof = 8 + 44 * (np.abs(x - 50) / 50) ** 1.5
HIGH, LOW = 32.0, 19.0
a.fill_between(x, -6, prof, color=SOIL, lw=0, zorder=1); a.plot(x, prof, color=SOIL_E, lw=1.4, zorder=3)
wet = prof < HIGH
a.fill_between(x[wet], prof[wet], HIGH, color=P["lblue"], alpha=0.85, lw=0, zorder=2)
for lev, ls_ in [(HIGH, "-"), (LOW, (0, (4, 2)))]:
    xs_ = x[prof < lev]; a.plot([xs_.min(), xs_.max()], [lev, lev], color=DK["blue"], lw=1.2, ls=ls_, zorder=4)
a.text(50, HIGH + 1.8, "full pool", fontsize=FS["small"], ha="center", va="bottom", color=DK["blue"])
a.text(50, LOW + 1.8, "lowest operating level", fontsize=FS["small"], ha="center", va="bottom", color=DK["blue"])
for side in (-1, 1):
    m = (prof >= LOW) & (prof <= HIGH) & ((x - 50) * side > 0)
    a.plot(x[m], prof[m], color=P["blue"], lw=5.0, solid_capstyle="butt", zorder=5)
    m = (prof > HIGH + 4) & (prof < HIGH + 17) & ((x - 50) * side > 0)
    a.plot(x[m], prof[m], color=P["teal"], lw=5.0, solid_capstyle="butt", zorder=5)
for xv, size in [(5.2, 1.5), (7.4, 1.7), (9.6, 1.6), (11.8, 1.8), (94.0, 1.6), (91.8, 1.9), (89.6, 1.7), (87.4, 1.9), (85.2, 1.6), (96.2, 1.5), (90.7, 1.4)]:
    yv = np.interp(xv, x, prof) + 2.0
    a.plot([xv, xv], [yv - 1.5, yv + 1.2], color="#7A9A78", lw=1.1, zorder=6); a.add_patch(Circle((xv, yv + 2.6), size, fc="#7DB483", ec="none", zorder=6))
sx, sy = 50, 70
a.add_patch(Rectangle((sx - 3.2, sy - 1.6), 6.4, 3.2, fc="#4A4A4A", ec="none", zorder=7))
for dx in (-8.8, 4.2): a.add_patch(Rectangle((sx + dx, sy - 0.9), 4.6, 1.8, fc=P["blue"], ec="none", zorder=7))
for tx in (12, 24, 76, 88):
    a.plot([sx, tx], [sy - 2, np.interp(tx, x, prof) + 4.5], color="#8A8A8A", lw=0.7, ls=(0, (1.5, 2.5)), zorder=2)
a.text(sx + 13, sy, "optical satellite", fontsize=FS["small"], va="center", ha="left", color="#4A4A4A")
def label(txt, xy, xyt, col, rel=(0.0, 0.5)):
    a.annotate(txt, xy=xy, xytext=xyt, fontsize=FS["ann"], fontweight="bold", color=col, ha="left", va="center", zorder=8,
               arrowprops=dict(arrowstyle="-", color=col, lw=1.0, shrinkA=2, shrinkB=1, relpos=rel))
label("Upland control (R)", (7.0, np.interp(7.0, x, prof) - 0.8), (1.5, 67), DK["teal"], rel=(0.2, 0.0))
a.annotate("Drawdown zone (S)", xy=(20.5, np.interp(20.5, x, prof) - 0.5), xytext=(20.5, 9.5), fontsize=FS["ann"], fontweight="bold", color=DK["blue"], ha="center", va="center", zorder=8,
           arrowprops=dict(arrowstyle="-", color=DK["blue"], lw=1.0, shrinkA=2, shrinkB=1))
a.add_patch(Rectangle((60, -3.5), 38.5, 17, fc=(1, 1, 1, 0.85), ec="#AAAAAA", lw=0.8, zorder=8))
a.text(79.2, 9.0, "Contrast", fontsize=FS["small"], ha="center", va="center", color="#555555", zorder=9)
a.text(79.2, 3.4, "G = S − R", fontsize=13, ha="center", va="center", color=C_G, fontweight="bold", zorder=9)
a.text(79.2, -1.6, "difference between trends", fontsize=FS["small"], ha="center", va="center", color="#555555", style="italic", zorder=9)
panel(a, "a", x=0.0, y=0.965)

# ======================= b: counterfactuals bracket the effect =======================
b = fig.add_subplot(top[1]); b.set_xlim(-0.03, 1.55); b.set_ylim(0.17, 1.0)
t = np.linspace(0, 1, 40)
S = 0.50 - 0.09 * t; Rr = 0.60 + 0.30 * t; cf_stable = 0.50 + 0 * t; cf_par = 0.50 + 0.30 * t
b.plot(t, cf_stable, color=C_S, lw=1.6, ls=(0, (4, 2))); b.plot(t, cf_par, color=C_S, lw=1.6, ls=(0, (1.2, 2)))
b.plot(t, Rr, color=C_R, lw=2.4); b.plot(t, S, color=C_S, lw=2.6)
xe = 1.03
b.text(xe + 0.17, Rr[-1], "R, upland control", color=DK["teal"], fontsize=FS["ann"], fontweight="bold", va="center")
b.text(xe + 0.17, cf_par[-1] - 0.025, "S if it followed R", color=DK["blue"], fontsize=FS["ann"], va="center")
b.text(xe + 0.17, cf_stable[-1] + 0.0, "S if stable", color=DK["blue"], fontsize=FS["ann"], va="center")
b.text(xe + 0.17, S[-1] - 0.0, "S, observed", color=DK["blue"], fontsize=FS["ann"], fontweight="bold", va="center")
xb1, xb2 = 1.0, 1.0
b.annotate("", xy=(xb1 + 0.012, S[-1] + 0.012), xytext=(xb1 + 0.012, cf_stable[-1] - 0.012), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0))
b.annotate("", xy=(xb1 + 0.095, S[-1] + 0.005), xytext=(xb1 + 0.095, cf_par[-1] - 0.012), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0))
b.text(0.0, 0.25, "Effect on the trend lies between\n$b_S$ (S stable) and $b_S - b_R$ (S follows R)", fontsize=FS["ann"], color=INK, ha="left", va="center", fontweight="bold", linespacing=1.3)
b.text(0.02, 0.965, "Observed S and R,\nand two counterfactuals for S", fontsize=FS["ann"], color="#444444", ha="left", va="top", linespacing=1.15)
b.set_xticks([]); b.set_yticks([]); b.spines[["left", "bottom"]].set_color("#999999"); b.spines[["top", "right"]].set_visible(False)
b.set_xlabel("time", fontsize=FS["small"], labelpad=1, loc="center", color="#666666"); b.set_ylabel("vegetation index", fontsize=FS["small"], labelpad=2, color="#666666")
panel(b, "b", x=-0.06, y=1.03)

# ======================= c: design table =======================
c = fig.add_subplot(gs[1]); c.set_xlim(0, 100); c.set_ylim(0, 41); c.axis("off")
rows = [("Sentinel-2, 2018–2025", "74 reservoirs · 502 windows", "How large are S, R and G?", "Figs 3, 4", P["blue"]),
        ("Landsat and MODIS, 2000–2025", "34 reservoirs (18 pre-impoundment)", "Was the upland trend stable?", "Fig. 5", P["teal"]),
        ("Year-to-year changes", "70 reservoirs · 8 years", "Does S follow R from year to year?", "Fig. 6", P["salmon"]),
        ("Remaining change", "3 initial-impoundment reservoirs · 6 lakes", "Where does the remaining decline occur?", "Fig. 7", P["amber"]),
        ("Daily storage", "49 reservoirs · 1.5 M pixel-years", "Does exposure explain greenness?", "Fig. 8", P["lblue"])]
c.text(0.5, 38.0, "Design: five lines of evidence", fontsize=FS["title"], fontweight="bold", va="center")
c.plot([0, 100], [35.0, 35.0], color=INK, lw=1.0)
for xx, s_, ha in [(1.0, "Evidence", "left"), (31.0, "Sites", "left"), (66.0, "Question", "left")]:
    c.text(xx, 32.8, s_, fontsize=FS["small"], color="#666666", va="center", ha=ha)
rh = 5.9
for i, (nm, sz, q, fg, col) in enumerate(rows):
    y = 28.4 - i * rh
    c.add_patch(Circle((2.0, y), 1.35, fc=col, ec="none")); c.text(2.0, y - 0.05, str(i + 1), fontsize=FS["small"], ha="center", va="center", fontweight="bold", color=INK)
    c.text(4.6, y, nm, fontsize=FS["ann"], fontweight="bold", va="center")
    c.text(31.0, y, sz, fontsize=FS["ann"], va="center", ha="left", color="#333333")
    c.text(66.0, y, q, fontsize=FS["small"] + 0.3, va="center", ha="left")
    c.plot([0, 100], [y - rh / 2, y - rh / 2], color="#DDDDDD", lw=0.6)
panel(c, "c", x=0.0, y=0.97)
align_panels(fig, rows=[[a, b]], cols=[[a, c]])
save(fig, "Fig01_concept")
