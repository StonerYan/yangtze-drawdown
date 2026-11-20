"""Fig. 7 - Where the decline that remains sits: first-filling reservoirs, occurrence bands, vegetated fraction and floodplain lakes."""
from figkit import *

M6 = json.load(open(f"{AN}/r6_misc.json", encoding="utf-8")); R7 = json.load(open(f"{AN}/r7_bands.json", encoding="utf-8")); VT = json.load(open(f"{AN}/vegfrac_thresholds.json", encoding="utf-8"))
fig = plt.figure(figsize=(W, 118 * MM))
gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.32, left=0.075, right=0.985, top=0.94, bottom=0.10)

# ---- a: first-filling reservoirs ----
a = fig.add_subplot(gs[0, 0]); FF = M6["first_filling_series"]
nm = {"Lower_Jinsha_Wudongde": "Wudongde", "Lower_Jinsha_Baihetan": "Baihetan", "Yalong_Lianghekou": "Lianghekou"}; cl = {"Lower_Jinsha_Wudongde": P["amber"], "Lower_Jinsha_Baihetan": P["red"], "Yalong_Lianghekou": P["blue"]}
dk = {"Lower_Jinsha_Wudongde": DK["amber"], "Lower_Jinsha_Baihetan": DK["red"], "Yalong_Lianghekou": DK["blue"]}
yrs = list(range(2017, 2026))
for k in nm:
    s = [FF[k][str(y)][0] for y in yrs]; r = [FF[k][str(y)][1] for y in yrs]
    a.plot(yrs, r, color=cl[k], lw=1.2, ls=(0, (3, 2)), alpha=0.9); a.plot(yrs, s, "-o", color=dk[k], lw=2.0, ms=3.8)
    a.plot([], [], "-o", color=dk[k], lw=2.0, ms=3.8, label=nm[k])
a.set_xticks(range(2017, 2026, 2)); a.set_ylabel("Peak EVI2"); a.set_ylim(-0.06, 0.56)
a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.01), fontsize=FS["small"] - 0.5, handlelength=1.6, labelspacing=0.18, borderpad=0.2)
ttl(a, "Three reservoirs at initial impoundment"); panel(a, "a", x=-0.17, y=1.08)

# ---- b: S trend by occurrence band ----
b = fig.add_subplot(gs[0, 1]); bands = ["15", "35", "55", "75"]; lab = ["15–35", "35–55", "55–75", "75–95"]
for i, k in enumerate(bands):
    d_ = R7["n71"][k]; b.bar(i, d_["mean"], color=C_S, width=0.62); b.errorbar(i, d_["mean"], yerr=[[d_["mean"] - d_["lo"]], [d_["hi"] - d_["mean"]]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=3)
    b.text(i, -12.6, f"{100 * d_['pixel_share']:.0f}%", ha="center", va="bottom", fontsize=FS["small"], color="#555")
b.text(-0.45, -12.6, "pixels:", ha="right", va="bottom", fontsize=FS["small"], color="#555")
b.axhline(0, color="#777", lw=0.8); b.set_xticks(range(4)); b.set_xticklabels(lab, fontsize=FS["tick"]); b.set_xlabel("Surface-water occurrence (%)", labelpad=1)
b.set_ylabel("S trend\n(EVI2 × 10$^{-3}$ yr$^{-1}$)"); b.set_ylim(-13, 5.0); b.set_xlim(-1.3, 3.6)
ttl(b, "Established reservoirs, by band"); panel(b, "b", x=-0.30, y=1.08)

# ---- c: vegetated fraction trend by band ----
c = fig.add_subplot(gs[1, 0]); labs = ["15-35", "35-55", "55-75", "75-95"]
for i, k in enumerate(labs):
    d_ = VT[f"0.2|{k}"]["n71"]; c.bar(i, d_["mean"], color=P["teal"], width=0.62); c.errorbar(i, d_["mean"], yerr=[[d_["mean"] - d_["lo"]], [d_["hi"] - d_["mean"]]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=3)
    c.text(i, -0.55, f"{100 * VT[f'0.2|{k}']['Vmean']:.0f}%", ha="center", va="top", fontsize=FS["small"], color="#555")
c.text(-0.45, -0.55, "mean V:", ha="right", va="top", fontsize=FS["small"], color="#555")
c.axhline(0, color="#777", lw=0.8); c.set_xticks(range(4)); c.set_xticklabels(["15–35", "35–55", "55–75", "75–95"], fontsize=FS["tick"]); c.set_xlabel("Surface-water occurrence (%)", labelpad=1)
c.set_ylabel("V trend\n(percentage points yr$^{-1}$)"); c.set_ylim(-1.0, 1.9); c.set_xlim(-1.3, 3.6)
c.tick_params(axis="x", pad=11)
ttl(c, "Vegetated fraction (V)"); panel(c, "c", x=-0.17, y=1.08)

# ---- d: lakes ----
d = fig.add_subplot(gs[1, 1]); BL = M6["lakes"]["by_lake"]; order = ["WET_Liangzi", "WET_Shengjin", "WET_Chaohu", "WET_Dongting", "WET_Poyang", "WET_Honghu"]
yp = np.arange(len(order) + 1)[::-1]
for i, k in enumerate(order + ["mean"]):
    for v, off, col, dk_ in [("S", 0.17, C_S, DK["blue"]), ("R", -0.17, C_R, DK["teal"])]:
        e = BL[k][v] if k != "mean" else M6["lakes"]["mean"][v]
        d.plot([e["lo"], e["hi"]], [yp[i] + off] * 2, color=col, lw=2.4, solid_capstyle="round"); d.plot(e["mean"], yp[i] + off, "o", color=dk_, ms=4.4)
d.axvline(0, color="#777", lw=0.8); d.axhline(0.5, color="#BBBBBB", lw=0.8, ls=(0, (3, 2)))
d.set_yticks(yp); d.set_yticklabels([k.replace("WET_", "") for k in order] + ["All six lakes"], fontsize=FS["tick"]); d.grid(axis="x", color=GRID, lw=0.6); d.set_axisbelow(True)
d.set_xlabel("Trend (EVI2 × 10$^{-3}$ yr$^{-1}$)", labelpad=1); d.set_xlim(-70, 20)
d.text(0.98, 0.98, "S", transform=d.transAxes, color=DK["blue"], fontweight="bold", fontsize=FS["big"], ha="right", va="top")
d.text(0.98, 0.87, "R", transform=d.transAxes, color=DK["teal"], fontweight="bold", fontsize=FS["big"], ha="right", va="top")
ttl(d, "Six floodplain lakes"); panel(d, "d", x=-0.40, y=1.08)
align_panels(fig, rows=[[a, b], [c, d]], cols=[[a, c], [b, d]])
save(fig, "Fig07_remaining")
