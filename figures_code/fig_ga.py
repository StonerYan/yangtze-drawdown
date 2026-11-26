"""Graphical abstract: difference-in-trends -> bracket on the effect -> where a steep decline occurs."""
from figkit import *
from matplotlib.patches import Rectangle

RS = pd.read_csv(f"{AN}/res18.csv"); J = json.load(open(f"{AN}/r1_core.json", encoding="utf-8")); M6 = json.load(open(f"{AN}/r6_misc.json", encoding="utf-8"))
A71 = J["A"]["n71"]
GW, GH = 200, 95
fig = plt.figure(figsize=(GW * MM, GH * MM))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, GW); bg.set_ylim(GH, 0); bg.axis("off")
def axmm(x, y, w, h, **kw): return fig.add_axes([x / GW, 1 - (y + h) / GH, w / GW, h / GH], **kw)

bg.text(GW / 2, 7, "Upland greening widens the apparent decline in drawdown-zone greenness at Yangtze reservoirs", fontsize=11, fontweight="bold", ha="center", va="center", color=INK)
bg.plot([3, GW - 3], [14, 14], color=INK, lw=1.2)
cols = [(3, 63, P["salmon"], DK["red"], "1", "Drawdown zone minus upland"), (69, 62, P["blue"], DK["blue"], "2", "Bounds on the effect"), (137, 60, P["teal"], DK["teal"], "3", "Where steep decline occurs")]
for x0, w, c1, c2, n, t in cols:
    bg.add_patch(Rectangle((x0, 17.5), w, 1.6, fc=c1, ec="none"))
    bg.text(x0, 25, n, fontsize=15, fontweight="bold", color=c2, va="center"); bg.text(x0 + 6, 25, t, fontsize=9.5, fontweight="bold", color=INK, va="center")
for x_ in (67, 134): bg.plot([x_, x_], [20, GH - 3], color="#D5D5D5", lw=0.8)

# ---- column 1: map ----
ax = axmm(3, 33, 63, 31, projection=PC); basemap(ax, [97.3, 122.4, 24.0, 35.4], res="50m", lw_river=0.5)
m0 = RS.G >= 0
ax.scatter(RS.lon[~m0], RS.lat[~m0], s=12, c=C_G, edgecolor="white", lw=0.4, transform=PC, zorder=4)
ax.scatter(RS.lon[m0], RS.lat[m0], s=12, c="#BBBBBB", edgecolor="white", lw=0.4, transform=PC, zorder=4)
bg.text(3, 78, f"{int((~m0).sum())} of 74", fontsize=17, fontweight="bold", color=DK["red"], va="center")
bg.text(3, 86.5, "reservoirs show a declining\ndifference between drawdown\nzone and upland (G < 0)", fontsize=FS["ann"], color="#333333", va="center", linespacing=1.3)

# ---- column 2: bracket ----
b = axmm(76, 29, 50, 24); S0, R0, G0 = (A71[k]["mean"] for k in "SRG")
b.bar(0, S0, color=C_S, width=0.66); b.bar(1, R0, color=C_R, width=0.66); b.bar(2, G0, color=C_G, width=0.66)
b.axhline(0, color="#777", lw=0.8); b.set_xticks([0, 1, 2]); b.set_xticklabels(["S", "R", "G"], fontsize=FS["label"], fontweight="bold"); b.set_yticks([])
b.spines[["left", "top", "right"]].set_visible(False); b.set_ylim(-9.5, 7.5)
for x_, v in [(0, S0), (1, R0), (2, G0)]:
    b.text(x_, v + (0.5 if v > 0 else -0.5), f"{v:+.1f}".replace("-", "−"), ha="center", va="bottom" if v > 0 else "top", fontsize=FS["ann"], fontweight="bold", color=INK)
bg.text(100.5, 62.5, "Effect on the trend lies between", fontsize=FS["small"], color="#555555", va="center", ha="center")
bg.text(69, 78, f"{S0:.1f} and {G0:.1f}".replace("-", "−"), fontsize=17, fontweight="bold", color=DK["blue"], va="center")
bg.text(69, 86.5, "depending on whether the drawdown\nzone would have followed the upland", fontsize=FS["ann"], color="#333333", va="center", linespacing=1.3)

# ---- column 3: first filling ----
c = axmm(146, 33, 49, 26); FF = M6["first_filling_series"]["Lower_Jinsha_Baihetan"]; yrs = list(range(2017, 2026))
s = np.array([FF[str(y)][0] for y in yrs]); r = np.array([FF[str(y)][1] for y in yrs])
c.fill_between(yrs, s, r, color=C_G, alpha=0.14, lw=0); c.plot(yrs, r, "-o", color=DK["teal"], ms=3.5, lw=1.8); c.plot(yrs, s, "-o", color=C_S, ms=3.5, lw=1.8)
c.set_xticks([2017, 2021, 2025]); c.tick_params(labelsize=FS["small"]); c.set_ylim(-0.05, 0.62); c.set_yticks([0, 0.3, 0.6])
c.text(2024.9, 0.10, "S", color=C_S, fontsize=FS["label"], fontweight="bold", ha="right"); c.text(2024.9, 0.56, "R", color=DK["teal"], fontsize=FS["label"], fontweight="bold", ha="right")
c.text(2016.9, 0.10, "Baihetan", fontsize=FS["ann"], fontweight="bold", ha="left", va="top"); c.set_xlim(2016.8, 2025.3)
a3 = J["A"]["n3"]["S"]
bg.text(137, 78, "3 of 74", fontsize=17, fontweight="bold", color=DK["teal"], va="center")
bg.text(137, 86.5, f"reservoirs at first filling drop steeply\n(S {a3['mean']:.0f}; the other 71: S {S0:.1f})".replace("-", "−"), fontsize=FS["ann"], color="#333333", va="center", linespacing=1.3)
save(fig, "GraphicalAbstract")
