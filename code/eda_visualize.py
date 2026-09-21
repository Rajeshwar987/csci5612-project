"""
eda_visualize.py  --  CSCI 5612 Project, Module 1 Part 1
Exploratory data analysis figures for the cleaned NHTSA data.

Colour handling follows a validated categorical palette: hues are assigned by
slot in fixed order and never cycled, single-series charts use one colour
rather than a value-ramp, magnitude uses a single-hue ramp, and the correlation
matrix uses a blue/red diverging pair with a neutral grey midpoint.

Outputs: figures/fig01_*.png ... figures/fig16_*.png
         figures/snapshot_raw_*.png, figures/snapshot_clean_*.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- validated categorical slots, used in fixed order -------------------
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = SLOT[0]
ACCENT = SLOT[1]

INK = "#16212e"
INK_SOFT = "#46586c"
GRID = "#e4e8ee"
SURFACE = "#ffffff"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#86b6ef", "#f0efec", "#f0a3a2", "#c22d2c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans",          # sans-serif: no serif on figures
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK_SOFT,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",                  # solid hairline grid, never dashed
})


def frame(ax, xgrid=False, ygrid=True):
    """Recessive chrome: hairline solid grid, only two spines."""
    ax.set_axisbelow(True)
    ax.grid(axis="y" if ygrid else "x", visible=ygrid or xgrid)
    if xgrid:
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"    {name}", flush=True)


# ------------------------------------------------------------------ load
print("loading cleaned data ...", flush=True)
c = pd.read_csv(CLEAN / "complaints_clean.csv.gz", low_memory=False,
                parse_dates=["date_incident", "date_filed"])
my = pd.read_csv(CLEAN / "vehicle_model_year.csv")
rec = pd.read_csv(CLEAN / "recalls_clean.csv", parse_dates=["date_reported"])
inv = pd.read_csv(CLEAN / "investigations_clean.csv.gz", low_memory=False,
                  parse_dates=["date_opened"])
print(f"  complaints {c.shape} | model-years {my.shape} | "
      f"recalls {rec.shape} | investigations {inv.shape}", flush=True)

print("building figures ...", flush=True)

# 1 -- complaint volume over time
s = c[c["report_year"] < 2026].groupby("report_year").size()
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(s.index, s.values, color=BLUE, linewidth=2, marker="o",
        markersize=5, markerfacecolor=BLUE, markeredgecolor=SURFACE,
        markeredgewidth=1.6)
peak = s.idxmax()
ax.annotate(f"peak: {s.max():,} in {peak}", xy=(peak, s.max()),
            xytext=(0, 12), textcoords="offset points", ha="center",
            fontsize=9.5, color=INK)
ax.set_title("Vehicle safety complaints filed per year, 2015-2025")
ax.set_xlabel("Year complaint was filed")
ax.set_ylabel("Complaints filed")
ax.set_ylim(bottom=0)
frame(ax)
save(fig, "fig01_complaints_per_year.png")

# 2 -- complaints by segment
s = c["segment"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(s.index, s.values, color=BLUE, height=0.68)
for i, v in enumerate(s.values):
    ax.text(v + s.max() * 0.012, i, f"{v:,}", va="center",
            fontsize=9.5, color=INK_SOFT)
ax.set_title("Complaint volume by vehicle segment")
ax.set_xlabel("Number of complaints")
ax.set_xlim(0, s.max() * 1.12)
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig02_complaints_by_segment.png")

# 3 -- harm level
s = c["harm_level"].value_counts().sort_values()
pct = 100 * s / s.sum()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(s.index, s.values, color=BLUE, height=0.68)
for i, (v, p) in enumerate(zip(s.values, pct.values)):
    ax.text(v + s.max() * 0.012, i, f"{v:,}  ({p:.1f}%)", va="center",
            fontsize=9.5, color=INK_SOFT)
ax.set_title("How harmful were the reported incidents?")
ax.set_xlabel("Number of complaints")
ax.set_xscale("log")
ax.set_xlim(1, s.max() * 6)
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig03_harm_level.png")

# 4 -- top components
s = c["component_primary"].value_counts().head(12).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([t.title() for t in s.index], s.values, color=BLUE, height=0.7)
ax.set_title("Twelve vehicle systems drawing the most complaints")
ax.set_xlabel("Number of complaints")
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig04_top_components.png")

# 5 -- crash rate by component (highlight the extreme, rest recessive)
g = (c.groupby("component_primary")
     .agg(n=("odiNumber", "count"), crash_rate=("crash", "mean")))
g = g[g["n"] >= 400].sort_values("crash_rate").tail(12)
colors = [ACCENT if i == len(g) - 1 else BLUE for i in range(len(g))]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([t.title() for t in g.index], 100 * g["crash_rate"], color=colors, height=0.7)
ax.text(100 * g["crash_rate"].iloc[-1] - 0.4, len(g) - 1,
        f"{100*g['crash_rate'].iloc[-1]:.0f}%", va="center", ha="right",
        fontsize=9.5, color=SURFACE, fontweight="bold")
ax.set_title("Share of complaints involving a crash, by vehicle system")
ax.set_xlabel("Complaints reporting a crash (%)")
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig05_crash_rate_by_component.png")

# 6 -- reporting lag
d = c.loc[c["days_to_report"] > 0, "days_to_report"]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(d, bins=np.logspace(0, np.log10(d.max()), 45), color=BLUE)
ax.axvline(d.median(), color=ACCENT, linewidth=2)
ax.text(d.median() * 1.15, ax.get_ylim()[1] * 0.88,
        f"median {d.median():.0f} days", color=INK, fontsize=9.5)
ax.set_xscale("log")
ax.set_title("Delay between an incident and the complaint being filed")
ax.set_xlabel("Days from incident to filing (log scale)")
ax.set_ylabel("Number of complaints")
frame(ax)
save(fig, "fig06_reporting_lag.png")

# 7 -- vehicle age at incident
s = c[c["vehicle_age_at_incident"] <= 12]["vehicle_age_at_incident"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(s.index, s.values, color=BLUE, width=0.72)
ax.set_title("Vehicle age when the reported problem occurred")
ax.set_xlabel("Vehicle age at incident (years)")
ax.set_ylabel("Number of complaints")
ax.set_xticks(range(0, 13))
frame(ax)
save(fig, "fig07_vehicle_age.png")

# 8 -- fire rate by segment
g = c.groupby("segment").agg(n=("odiNumber", "count"), fire_rate=("fire", "mean"))
g = g[g["n"] >= 300].sort_values("fire_rate")
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(g.index, 100 * g["fire_rate"], color=BLUE, height=0.66)
for i, v in enumerate(100 * g["fire_rate"]):
    ax.text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=9.5, color=INK_SOFT)
ax.set_title("Share of complaints that involved a fire, by segment")
ax.set_xlabel("Complaints reporting a fire (%)")
ax.set_xlim(0, 100 * g["fire_rate"].max() * 1.25)
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig08_fire_rate_by_segment.png")

# 9 -- star rating distribution
s = my["OverallRating"].round(1).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([str(i) for i in s.index], s.values, color=BLUE, width=0.66)
ax.set_title("Distribution of NHTSA overall crash-test ratings")
ax.set_xlabel("Overall star rating (averaged across rated body styles)")
ax.set_ylabel("Model-years")
frame(ax)
save(fig, "fig09_star_rating_distribution.png")

# 10 -- rollover probability by segment
order = my.groupby("segment")["RolloverPossibility"].median().sort_values().index
data = [my.loc[my["segment"] == s_, "RolloverPossibility"] * 100 for s_ in order]
fig, ax = plt.subplots(figsize=(8, 4.4))
bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                medianprops=dict(color=SURFACE, linewidth=1.8),
                flierprops=dict(marker="o", markersize=4,
                                markerfacecolor=INK_SOFT, markeredgecolor="none",
                                alpha=0.5))
for patch in bp["boxes"]:
    patch.set(facecolor=BLUE, edgecolor="none")
for w in bp["whiskers"] + bp["caps"]:
    w.set(color=INK_SOFT, linewidth=1)
ax.set_yticklabels(order)
ax.set_title("Rollover probability by vehicle segment")
ax.set_xlabel("NHTSA estimated rollover probability (%)")
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig10_rollover_by_segment.png")

# 11 -- crash rate vs rollover probability (single series: all-pairs cap)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(my["RolloverPossibility"] * 100, my["crash_rate"] * 100,
           s=np.clip(my["n_complaints"] / 6, 14, 220), color=BLUE,
           alpha=0.55, edgecolor=SURFACE, linewidth=1.4)
for _, r in my.nlargest(3, "crash_rate").iterrows():
    ax.annotate(f"{r['make'].title()} {r['model'].title()} {int(r['model_year'])}",
                xy=(r["RolloverPossibility"] * 100, r["crash_rate"] * 100),
                xytext=(6, 7), textcoords="offset points", fontsize=9, color=INK)
ax.set_title("Crash-involved complaints against rollover probability")
ax.set_xlabel("Rollover probability (%)")
ax.set_ylabel("Complaints involving a crash (%)")
ax.text(0.99, 0.02, "marker size = complaint volume", transform=ax.transAxes,
        ha="right", fontsize=9, color=INK_SOFT, style="italic")
frame(ax)
ax.grid(axis="x", visible=True)
save(fig, "fig11_crashrate_vs_rollover.png")

# 12 -- correlation heatmap (diverging, neutral midpoint)
num = my[["n_complaints", "crash_rate", "fire_rate", "pct_severe",
          "injuries_per_100_complaints", "median_days_to_report",
          "mean_vehicle_age", "n_recalls", "OverallRating",
          "RolloverPossibility", "InvestigationCount"]]
corr = num.corr()
labels = ["Complaints", "Crash rate", "Fire rate", "% severe", "Injuries/100",
          "Days to report", "Mean age", "Recalls", "Star rating",
          "Rollover prob.", "Investigations"]
fig, ax = plt.subplots(figsize=(8.4, 7))
im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)), labels)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color=SURFACE if abs(v) > 0.55 else INK)
ax.set_title("Correlations among model-year safety measures")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Pearson correlation", color=INK_SOFT)
cb.outline.set_visible(False)
ax.grid(False)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
save(fig, "fig12_correlation_heatmap.png")

# 13 -- most-complained model-years
top = my.nlargest(15, "n_complaints").sort_values("n_complaints")
lab = [f"{r.make.title()} {r.model.title()} {int(r.model_year)}"
       for r in top.itertuples()]
fig, ax = plt.subplots(figsize=(8, 5.4))
ax.barh(lab, top["n_complaints"], color=BLUE, height=0.7)
ax.set_title("Fifteen model-years with the most complaints")
ax.set_xlabel("Number of complaints")
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig13_top_model_years.png")

# 14 -- harm-level mix by segment (stacked; slots in fixed order, 2px gaps)
levels = ["No crash or fire", "Crash, no injury", "Fire, no crash", "Injury", "Fatal"]
ct = pd.crosstab(c["segment"], c["harm_level"], normalize="index") * 100
ct = ct.reindex(columns=[l for l in levels if l in ct.columns]).fillna(0)
ct = ct.loc[ct["No crash or fire"].sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
left = np.zeros(len(ct))
for k, col in enumerate(ct.columns):
    ax.barh(ct.index, ct[col], left=left, color=SLOT[k], height=0.66,
            label=col, edgecolor=SURFACE, linewidth=2)   # 2px surface gap
    left += ct[col].values
ax.set_title("Outcome mix of complaints within each segment")
ax.set_xlabel("Share of that segment's complaints (%)")
ax.set_xlim(0, 100)
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28))
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig14_harm_mix_by_segment.png")

# 15 -- investigations opened per year (bulk-download source)
s = inv[inv["date_opened"].dt.year.between(1995, 2025)]
s = s.groupby(s["date_opened"].dt.year).size()
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.fill_between(s.index, s.values, color=BLUE, alpha=0.18)
ax.plot(s.index, s.values, color=BLUE, linewidth=2)
ax.set_title("Defect investigation records opened per year (bulk ODI file)")
ax.set_xlabel("Year investigation record was opened")
ax.set_ylabel("Records opened")
ax.set_ylim(bottom=0)
frame(ax)
save(fig, "fig15_investigations_per_year.png")

# 16 -- recall components
s = rec["component_primary"].value_counts().head(12).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([t.title() for t in s.index], s.values, color=BLUE, height=0.7)
ax.set_title("Vehicle systems most often subject to a safety recall")
ax.set_xlabel("Number of recall records")
frame(ax, xgrid=True, ygrid=False)
save(fig, "fig16_recall_components.png")


# ------------------------------------------------------- raw/clean snapshots
def snapshot(df, cols, title, fname, nrows=6, width=13.5):
    """Render a small table image so readers can see the data itself."""
    sub = df[cols].head(nrows).copy()
    for col in sub.columns:
        sub[col] = sub[col].astype(str).str.slice(0, 26)
    fig, ax = plt.subplots(figsize=(width, 0.42 * nrows + 1.25))
    ax.axis("off")
    t = ax.table(cellText=sub.values, colLabels=[c[:22] for c in sub.columns],
                 cellLoc="left", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.6)
    t.scale(1, 1.42)
    for (row, _), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#f5f7fa")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK_SOFT)
    ax.set_title(title, fontsize=11.5, fontweight="semibold", color=INK, pad=14)
    save(fig, fname)


raw_c = pd.read_csv(RAW / "complaints_raw.csv.gz", nrows=6, low_memory=False)
snapshot(raw_c, ["odiNumber", "manufacturer", "crash", "fire", "numberOfInjuries",
                 "dateOfIncident", "vin", "components", "summary"],
         "RAW complaints as returned by the NHTSA API",
         "snapshot_raw_complaints.png")

snapshot(c, ["odiNumber", "make", "model", "model_year", "segment", "harm_level",
             "component_primary", "days_to_report", "vehicle_age_at_incident", "wmi"],
         "CLEANED complaints: typed, de-duplicated, derived fields added",
         "snapshot_clean_complaints.png")

# Show rows that actually exhibit the defects the caption describes, rather
# than the first six rows, which happen to be clean.
raw_r = pd.read_csv(RAW / "safety_ratings_raw.csv.gz", low_memory=False)
flawed = raw_r[(raw_r["RolloverPossibility"] == 0.0) |
               (raw_r["OverallRating"] == "Not Rated")]
snapshot(flawed, ["Make", "Model", "ModelYear", "OverallRating", "RolloverRating",
                  "RolloverPossibility", "RolloverRating2", "dynamicTipResult"],
         "RAW safety ratings: 'Not Rated' text and 0.0 rollover sentinels",
         "snapshot_raw_ratings.png", width=11)

snapshot(my, ["make", "model", "model_year", "segment", "n_complaints",
              "crash_rate", "OverallRating", "RolloverPossibility",
              "rollover_risk_band", "rating_class"],
         "CLEANED model-year table: one row per vehicle, ready for modelling",
         "snapshot_clean_model_year.png")

print(f"\ndone - {len(list(FIG.glob('*.png')))} images in {FIG}")
