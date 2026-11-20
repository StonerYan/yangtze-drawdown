"""Fig. 2 - Study system: 441-reservoir inventory, sampling funnel, zoom maps and measurement window."""
import os
from figkit import *
from matplotlib.lines import Line2D
from shapely.geometry import MultiPoint

Y = pd.read_csv(f"{DR}/ywr_reservoir.csv").merge(pd.read_csv(f"{DD}/ywr_meta.csv"), on="GRAND_ID", how="left")
WT = pd.read_parquet(f"{D2}/win_table.parquet"); RT = WT.groupby("reservoir").agg(lon=("wlon", "mean"), lat=("wlat", "mean"), n=("window", "size"))
LS = sorted({f.rsplit("_w", 1)[0] for f in os.listdir(f"{D2}/landsat_obs")}); RL = RT.loc[[r for r in LS if r in RT.index]]
WW = pd.read_parquet(f"{D2}/wet_win_table.parquet").groupby("reservoir")[["lon", "lat"]].mean()
DAILY = set(pd.read_csv(f"{AN}/exposure_validation_all.csv").reservoir)          # 52 reservoirs with daily storage
FIRST = ["Yalong_Lianghekou", "Lower_Jinsha_Baihetan", "Lower_Jinsha_Wudongde"]
RT["daily"] = RT.index.isin(DAILY); RT["first"] = RT.index.isin(FIRST)
def sym(axx, df, k=1.0, z=5):
    d_ = df[df.daily]; o_ = df[~df.daily]; s_ = lambda q: 22 * k + 4.5 * k * q.n
    axx.scatter(d_.lon, d_.lat, s=s_(d_), facecolor=P["lblue"], edgecolor=INK, lw=0.8, alpha=0.9, transform=PC, zorder=z)
    axx.scatter(o_.lon, o_.lat, s=s_(o_), facecolor="none", edgecolor=INK, lw=0.8, transform=PC, zorder=z)
    f_ = df[df["first"]]; axx.scatter(f_.lon, f_.lat, s=60 * k, marker="*", facecolor=DK["red"], edgecolor="white", lw=0.5, transform=PC, zorder=z + 1)
cnt = Y.type.value_counts()

fig = plt.figure(figsize=(W, 170 * MM))
gs = fig.add_gridspec(3, 1, height_ratios=[84, 16, 62], hspace=0.07, left=0.045, right=0.985, top=0.965, bottom=0.01)
EXT = [97.3, 122.4, 24.0, 35.4]
ax = fig.add_subplot(gs[0], projection=PC); basemap(ax, EXT, res="10m"); geo_ticks(ax, [100, 105, 110, 115, 120], [25, 30, 35])
# Inventory points plus the Sentinel-2 and lake symbols, so the outline encloses every plotted site.
pts = list(zip(Y.lon, Y.lat)) + list(zip(RT.lon, RT.lat)) + list(zip(WW.lon, WW.lat))
hull = MultiPoint(pts).convex_hull.buffer(0.5)
ax.add_geometries([hull], PC, facecolor="none", edgecolor="#7A7A7A", lw=0.8, linestyle=(0, (4, 3)), zorder=2)
for t in ["split", "single", "counter-seasonal"]:
    m = Y.type == t; ax.scatter(Y.lon[m], Y.lat[m], s=9, c=TYPE_C[t], alpha=0.9, lw=0, transform=PC, zorder=3)
sym(ax, RT, 1.0, 5)
ax.scatter(RL.lon, RL.lat, s=11, marker="s", c=INK, lw=0, transform=PC, zorder=6)
ax.scatter(WW.lon, WW.lat, s=55, marker="D", c=WET_C, edgecolor=INK, lw=0.8, transform=PC, zorder=7)
zA = [99.2, 104.6, 25.3, 31.3]; zB = [111.2, 119.2, 28.3, 32.4]
box_on(ax, zA, DK["red"], 1.4); box_on(ax, zB, DK["red"], 1.4)
for nm, ext in [("b", zA), ("c", zB)]:
    ax.text(ext[1] - 0.30, ext[2] + 0.48, nm, fontsize=FS["big"], fontweight="bold", color="white",
            transform=PC, ha="right", va="bottom", zorder=12,
            bbox=dict(boxstyle="square,pad=0.18", fc=DK["red"], ec="none"))
for nm, x, y, xt, yt, ha in [("Three Gorges", 110.95, 30.85, 109.15, 34.55, "right"), ("Danjiangkou", 111.5, 32.55, 112.45, 34.35, "left")]:
    ax.scatter([x], [y], s=18, c=INK, edgecolor="white", lw=0.6, transform=PC, zorder=10)
    ax.annotate(nm, xy=(x, y), xytext=(xt, yt), fontsize=FS["ann"], color=INK, ha=ha, va="center", transform=PC, zorder=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkB=2, shrinkA=2), bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.0))
ax.text(121.9, 24.35, "East China Sea", fontsize=FS["small"], color="#5F8399", ha="right", style="italic", transform=PC)
# north arrow at the upper right; scale bar centred on the same vertical line
NX, TIP, ASZ = 0.955, 0.900, 0.082
north_arrow(ax, NX, TIP, ASZ)
lon_c, lat0 = ax.transData.inverted().transform(ax.transAxes.transform((NX, TIP - ASZ - 0.070)))
dlon = 100 / (111.32 * np.cos(np.radians(lat0)))
scale_bar_deg(ax, lon_c - dlon / 2, lat0, 100)
ttl(ax, "Yangtze River Basin reservoir network"); panel(ax, "a", x=-0.045, y=1.035)

# inset donut: operation calendars
ie = ax.inset_axes([0.012, 0.635, 0.145, 0.345]); ie.patch.set_alpha(0)
donut(ie, [cnt["counter-seasonal"], cnt["single"], cnt["split"]], [TYPE_C["counter-seasonal"], TYPE_C["single"], TYPE_C["split"]], labels=["", "", ""], fs=6.0, width=0.38)
ie.text(0, 0.11, "441", ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK); ie.text(0, -0.20, "reservoirs", ha="center", va="center", fontsize=5.5, color=INK)
ie.set_xlim(-1.05, 1.05); ie.set_ylim(-1.05, 1.05); ie.set_aspect("equal")

# legend row
LA = fig.add_subplot(gs[1]); LA.axis("off")
H = [Line2D([], [], marker="o", ls="", ms=6, mfc=TYPE_C[t], mec="none", label=f"{t.capitalize()} calendar (n = {int((Y.type == t).sum())})") for t in ["counter-seasonal", "single", "split"]]
H += [Line2D([], [], marker="o", ls="", ms=9, mfc=P["lblue"], mec=INK, mew=1.0, label="Sentinel-2, daily storage (n = 52)"),
      Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK, mew=1.0, label="Sentinel-2, random sample (n = 22)"),
      Line2D([], [], marker="*", ls="", ms=10, mfc=DK["red"], mec="white", mew=0.5, label="First filling in 2020–2023 (n = 3)"),
      Line2D([], [], marker="s", ls="", ms=5.5, mfc=INK, mec="none", label="Landsat 2000–2025 subset (n = 34)"),
      Line2D([], [], marker="D", ls="", ms=7, mfc=WET_C, mec=INK, mew=0.8, label="Floodplain lake (n = 6)")]
LA.legend(handles=H, loc="center", ncol=3, fontsize=FS["ann"], handletextpad=0.4, labelspacing=0.7, columnspacing=2.2)

# ---------- bottom row ----------
bot = gs[2].subgridspec(1, 3, width_ratios=[0.95, 1.95, 1.0], wspace=0.28)

# b: cascade zoom
b = fig.add_subplot(bot[0], projection=PC); basemap(b, zA, res="10m", lw_river=0.8); geo_ticks(b, [100, 102, 104], [26, 28, 30])
inb = lambda df: df[(df.lon.between(*zA[:2])) & (df.lat.between(*zA[2:]))]
yy = inb(Y)
for t in ["split", "single", "counter-seasonal"]:
    m = yy.type == t; b.scatter(yy.lon[m], yy.lat[m], s=14, c=TYPE_C[t], alpha=0.95, lw=0, transform=PC, zorder=3)
rb = inb(RT); sym(b, rb, 1.3, 5)
lab = {"Lower_Jinsha_Baihetan": ("Baihetan", 0.2, 0.0, "left", "center"), "Lower_Jinsha_Wudongde": ("Wudongde", 0.0, -0.22, "center", "top"),
       "Yalong_Lianghekou": ("Lianghekou", 0.2, 0.0, "left", "center"), "Lower_Jinsha_Xiluodu": ("Xiluodu", -0.2, 0.0, "right", "center"),
       "Lower_Jinsha_Xiangjiaba": ("Xiangjiaba", -0.2, 0.15, "right", "bottom")}
for k, (nm, dx, dy, ha, va) in lab.items():
    x, y = RT.loc[k, "lon"], RT.loc[k, "lat"]; stroke_text(b, x + dx, y + dy, nm, fontsize=FS["small"], ha=ha, va=va, transform=PC, zorder=9, color=INK)
ttl(b, "Upper cascades", pad=4); panel(b, "b", x=-0.30, y=1.07)

# c: floodplain lakes zoom
c = fig.add_subplot(bot[1], projection=PC); basemap(c, zB, res="10m", lw_river=0.8); geo_ticks(c, [112, 114, 116, 118], [29, 31])
yy = Y[(Y.lon.between(*zB[:2])) & (Y.lat.between(*zB[2:]))]
for t in ["split", "single", "counter-seasonal"]:
    m = yy.type == t; c.scatter(yy.lon[m], yy.lat[m], s=14, c=TYPE_C[t], alpha=0.95, lw=0, transform=PC, zorder=3)
rc = RT[(RT.lon.between(*zB[:2])) & (RT.lat.between(*zB[2:]))]; sym(c, rc, 1.3, 5)
c.scatter(WW.lon, WW.lat, s=70, marker="D", c=WET_C, edgecolor=INK, lw=0.8, transform=PC, zorder=7)
off = {"WET_Poyang": (0.05, -0.38, "center", "top"), "WET_Dongting": (0.0, -0.38, "center", "top"), "WET_Honghu": (-0.18, 0.28, "center", "bottom"), "WET_Liangzi": (0.0, 0.28, "center", "bottom"),
       "WET_Shengjin": (0.2, -0.38, "left", "top"), "WET_Chaohu": (-0.18, 0.28, "right", "bottom")}
for k, (dx, dy, ha, va) in off.items():
    x, y = WW.loc[k, "lon"], WW.loc[k, "lat"]; stroke_text(c, x + dx, y + dy, k.replace("WET_", ""), fontsize=FS["small"], ha=ha, va=va, transform=PC, zorder=9, color=DK["amber"], fontweight="bold")
ttl(c, "Floodplain lakes", pad=4); panel(c, "c", x=-0.12, y=1.07)

# d: measurement window
d = fig.add_subplot(bot[2])
PX = pd.read_parquet(f"{DC}/pixels.parquet"); p = PX[PX.window == "Han_Danjiangkou_w3"]
lat0 = p.lat.mean(); xk = (p.lon - p.lon.mean()) * 111.32 * np.cos(np.radians(lat0)); yk = (p.lat - lat0) * 110.57
cat = np.select([p.cls == "dz", p.dist.between(60, 300), p.dist.between(300, 600), p.dist > 600], ["dz", "r1", "r2", "r3"], "x")
cols = {"dz": C_S, "r1": C_R, "r2": "#B9DFDB", "r3": "#DCEEEC", "x": "#F2F2F2"}
for k in ["r3", "r2", "x", "r1", "dz"]:
    mm_ = cat == k; d.scatter(xk[mm_], yk[mm_], s=2.6, c=cols[k], lw=0, rasterized=True, marker="s")
d.set_aspect("equal"); d.set_xlim(-2.05, 2.05); d.set_ylim(-2.05, 2.05); d.set_xticks([-2, 0, 2]); d.set_yticks([-2, 0, 2])
d.set_xlabel("km", labelpad=1); d.set_ylabel("km", labelpad=0); d.spines[["top", "right"]].set_visible(True)
ttl(d, "Danjiangkou window", pad=4)
hd = [Line2D([], [], marker="s", ls="", ms=6, mfc=cols["dz"], mec="none", label="Drawdown zone (S)"), Line2D([], [], marker="s", ls="", ms=6, mfc=cols["r1"], mec="none", label="Upland control (R), 60–300 m"),
      Line2D([], [], marker="s", ls="", ms=6, mfc=cols["r2"], mec="none", label="300–600 m"), Line2D([], [], marker="s", ls="", ms=6, mfc=cols["r3"], mec="#BBBBBB", mew=0.4, label="600–1,050 m")]
d.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1, fontsize=FS["small"], handletextpad=0.3, labelspacing=0.25)
panel(d, "d", x=-0.32, y=1.07)
align_panels(fig, rows=[[b, c, d]], cols=[[ax, b]])
save(fig, "Fig02_study_area")
