"""Figure helpers: maps, donuts, annotations."""
import numpy as np, pandas as pd, cartopy.crs as ccrs, cartopy.feature as cf
from matplotlib.patches import Rectangle, ConnectionPatch, FancyBboxPatch, FancyArrowPatch, Polygon, Wedge, Circle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from common import *

PC = ccrs.PlateCarree()
LAND, OCEAN, WATER, WATER_EDGE = "#F6F4EE", "#E6F0F5", "#CFE6F0", "#8EC1D8"


def basemap(ax, extent, res="10m", lw_river=0.6, coast=True):
    ax.set_extent(extent, PC)
    ax.add_feature(cf.NaturalEarthFeature("physical", "ocean", res), facecolor=OCEAN, edgecolor="none", zorder=0)
    ax.add_feature(cf.NaturalEarthFeature("physical", "land", res), facecolor=LAND, edgecolor="none", zorder=0.5)
    ax.add_feature(cf.NaturalEarthFeature("physical", "rivers_lake_centerlines", res), facecolor="none", edgecolor=WATER_EDGE, lw=lw_river, zorder=1)
    ax.add_feature(cf.NaturalEarthFeature("physical", "lakes", res), facecolor=WATER, edgecolor=WATER_EDGE, lw=0.3, zorder=1.1)
    if coast: ax.add_feature(cf.NaturalEarthFeature("physical", "coastline", res), facecolor="none", edgecolor="#9DB5C2", lw=0.5, zorder=1.2)
    for s in ax.spines.values(): s.set_linewidth(0.8); s.set_color(INK)


def geo_ticks(ax, xt, yt, size=None):
    size = size or FS["tick"]
    ax.set_xticks(xt, crs=PC); ax.set_yticks(yt, crs=PC)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°")); ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(labelsize=size, length=3, width=0.7, pad=2)


def box_on(ax, ext, color=INK, lw=1.2, z=9, ls="-"):
    ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False, ec=color, lw=lw, ls=ls, transform=PC, zorder=z))


def link_boxes(fig, ax_main, ext, ax_child, color="#666666", lw=0.7):
    """Thin connector lines from the zoom box on ax_main to the top corners of ax_child."""
    x0, x1, y0, y1 = ext
    for (px, py), (cx, cy) in [((x0, y0), (0, 1)), ((x1, y0), (1, 1))]:
        con = ConnectionPatch(xyA=(px, py), coordsA=PC._as_mpl_transform(ax_main), xyB=(cx, cy), coordsB=ax_child.transAxes, color=color, lw=lw, ls=(0, (3, 2)), zorder=0)
        fig.add_artist(con)


def donut(ax, vals, colors, labels=None, width=0.38, start=90, centre=None, fs=None, pct=True, edge="white", counter=True, min_share=0.06):
    fs = fs or FS["ann"]
    vals = np.asarray(vals, float); tot = vals.sum()
    w, _ = ax.pie(vals, colors=colors, startangle=start, counterclock=not counter, wedgeprops=dict(width=width, edgecolor=edge, lw=1.2), radius=1)
    ax.set_aspect("equal")
    if labels is not None:
        for wi, v in zip(w, vals):
            ang = np.deg2rad((wi.theta1 + wi.theta2) / 2); r = 1 - width / 2
            if v / tot > min_share: ax.text(r * np.cos(ang), r * np.sin(ang), (f"{100 * v / tot:.0f}%" if pct else f"{int(v)}"), ha="center", va="center", fontsize=fs, color="white", fontweight="bold")
    if centre: ax.text(0, 0, centre, ha="center", va="center", fontsize=fs + 1, color=INK, linespacing=1.15)
    return w


def north_arrow(ax, x=0.94, y=0.88, size=0.09):
    ax.annotate("", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=9))
    ax.text(x, y + 0.012, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=FS["small"], fontweight="bold")


def scale_bar_deg(ax, lon0, lat0, km=200, fs=None):
    fs = fs or FS["small"]
    dlon = km / (111.32 * np.cos(np.radians(lat0)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color=INK, lw=2.0, transform=PC, solid_capstyle="butt", zorder=10)
    ax.text(lon0 + dlon / 2, lat0 + 0.15, f"{km} km", transform=PC, ha="center", va="bottom", fontsize=fs, zorder=10)


def stroke_text(ax, x, y, s, **kw):
    import matplotlib.patheffects as pe
    t = ax.text(x, y, s, **kw); t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")]); return t


def pill(ax, x, y, s, fc="#FFFFFF", ec=INK, fs=None, pad=0.25, lw=0.7, tc=INK, **kw):
    return ax.text(x, y, s, fontsize=fs or FS["ann"], ha="center", va="center", color=tc, bbox=dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=lw), **kw)
