"""Shared paths, colour palette, typography, and statistics helpers for figure generation."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AN = os.path.join(ROOT, "results")
D2 = os.path.join(ROOT, "data", "cee2")
DC = os.path.join(ROOT, "data", "cee")
DR = os.path.join(ROOT, "data", "derived", "results")
DD = os.path.join(ROOT, "data", "derived")
OUTD = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- palette (reference image) ----
P = dict(teal="#8DCEC8", lblue="#ADD3E2", blue="#3480B8", amber="#FFBE7A", salmon="#FA8878", red="#C82423")
DK = dict(teal="#4E9C95", blue="#1F5C8A", red="#8F1A1A", amber="#D98F2B", salmon="#D4604F", grey="#5B5B5B")
C_S, C_R, C_G = P["blue"], P["teal"], P["red"]            # drawdown zone / upland reference / relative metric
CLS = {"sign-flipped": P["red"], "reference-dominated": P["salmon"], "genuine-dominated": P["blue"], "no decline": P["lblue"]}
CLS_ORDER = ["sign-flipped", "reference-dominated", "genuine-dominated", "no decline"]
TYPE_C = {"counter-seasonal": P["salmon"], "single": P["blue"], "split": P["teal"]}
WET_C = P["amber"]
DIV = LinearSegmentedColormap.from_list("div", [P["blue"], P["lblue"], "#F4F4F2", P["salmon"], P["red"]])
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F8F8", P["teal"], P["lblue"], P["blue"]])
WARM = LinearSegmentedColormap.from_list("warm", ["#FFF6E8", P["amber"], P["salmon"], P["red"]])
MM = 1 / 25.4
INK = "#2B2B2B"; GRID = "#E3E3E3"

W = 180 * MM                                         # full-page width (18 cm)
FS = dict(tick=8.5, label=9.5, ann=8.5, small=8, title=9.5, panel=14, big=11)   # minimum text size on any figure is 8 pt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 9, "axes.titlesize": FS["title"], "axes.labelsize": FS["label"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["ann"],
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2, "legend.frameon": False, "lines.linewidth": 1.5, "savefig.dpi": 600,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.titlepad": 5})


def panel(ax, s, x=-0.14, y=1.04, size=None):
    t = ax.text(x, y, s, transform=ax.transAxes, fontsize=size or FS["panel"], fontweight="bold", va="bottom", ha="left", color=INK, clip_on=False, zorder=20)
    t.set_gid("panel"); t._panel_ax = ax
    return t


def align_panels(fig, rows=(), cols=(), xgap_mm=1.4, ygap_mm=1.4):
    """Top-align panel letters within each row and left-align them within each column.

    A column is left-aligned only when the shared left edge stays outside the other
    panels. Letters that would otherwise land on a neighbouring panel keep the
    corner of their own axes.
    """
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fw, fh = fig.get_size_inches()
    xgap, ygap = xgap_mm * MM / fw, ygap_mm * MM / fh
    letter_w, letter_h = 4.0 * MM / fw, 5.0 * MM / fh

    letters = {}
    for t in fig.findobj(lambda o: getattr(o, "get_gid", lambda: None)() == "panel"):
        ax = getattr(t, "_panel_ax", None)
        if ax is not None:
            letters[ax] = t

    def ext_of(artist):
        try:
            e = artist.get_window_extent(ren)
        except Exception:
            return None
        if e.width <= 0 and e.height <= 0:
            return None
        x0, y0 = inv.transform((e.x0, e.y0)); x1, y1 = inv.transform((e.x1, e.y1))
        return x0, y0, x1, y1

    def top_of(ax):
        # Row height follows the axes spine. A legend or an inset above only one
        # panel must not lift the letters of the whole row.
        return ax.get_position().y1

    # Sit just outside the top-left corner. Chasing tick labels pulls a letter
    # across the gutter into the previous panel.
    nat = {ax: [ax.get_position().x0 - letter_w - xgap, top_of(ax) + ygap] for ax in letters}
    for row in rows:
        row = [ax for ax in row if ax in nat]
        if len(row) >= 2:
            y = max(nat[ax][1] for ax in row)
            for ax in row:
                nat[ax][1] = y

    children = {ch for p in fig.axes for ch in list(getattr(p, "child_axes", []) or [])}
    mains = [ax for ax in fig.axes if ax not in children and ax.get_position().width > 0.08 and ax.get_position().height > 0.08]
    for col in cols:
        col = [ax for ax in col if ax in nat]
        if len(col) < 2:
            continue
        x = min(nat[ax][0] for ax in col)
        blocked = False
        for other in mains:
            if other in col:
                continue
            ob = other.get_position()
            if not (x + letter_w > ob.x0 + 0.6 * MM / fw and x < ob.x1 - 0.01):
                continue
            for ax in col:
                ly = nat[ax][1]
                if ly < ob.y1 and ly + letter_h > ob.y0:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            for ax in col:
                nat[ax][0] = x

    for ax, t in letters.items():
        t.set_transform(fig.transFigure); t.set_ha("left"); t.set_va("bottom")
        t.set_position((nat[ax][0], nat[ax][1])); t.set_clip_on(False); t.set_zorder(30)


def ttl(ax, s, pad=5, **kw):
    """Left-aligned panel title."""
    ax.set_title(s, loc="left", fontsize=FS["title"], fontweight="bold", pad=pad, **kw)


def save(fig, name, pad=0.03):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", pad_inches=pad, dpi=300)
    plt.close(fig); print("saved", name)


# ---- statistics ----
def ts_slope(y, x=None):
    y = np.asarray(y, float); x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    m = np.isfinite(y)
    if m.sum() < 6: return np.nan
    return theilslopes(y[m], x[m])[0]


def boot_mean(v, B=4000, seed=1, stat=np.mean):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (B, len(v))); bs = stat(v[idx], axis=1)
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_res():
    """Reservoir-level table (74 reservoirs) with S, R, G trends and decomposition class."""
    sp = pd.read_csv(f"{D2}/res_spatial.csv"); rt = pd.read_parquet(f"{D2}/res_table.parquet").reset_index()
    cols = [c for c in rt.columns if c not in sp.columns or c == "reservoir"]
    return sp.merge(rt[cols], on="reservoir", how="left")


def classify(S, R):
    G = S - R
    return np.where(G >= 0, "no decline", np.where(S >= 0, "sign-flipped", np.where(R >= -S, "reference-dominated", "genuine-dominated")))
