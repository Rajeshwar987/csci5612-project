"""
pca_analysis.py  --  CSCI 5612 Project, Module 2 Part 2
Principal Component Analysis of the model-year vehicle safety table.

PCA needs the same shape of input as clustering: a purely numeric matrix with
no labels, standardised so that every variable contributes on equal terms.
The eigenvectors of the correlation matrix give the new axes and the
eigenvalues give the variance carried along each one.

Produces:
    data/clean/pca_sample.csv              the standardised input matrix
    data/clean/pca_components.csv          eigenvectors (loadings)
    data/clean/pca_scores.csv              vehicles in component space
    data/clean/pca_eigenvalues.csv         eigenvalue table
    figures/fig23..fig27                   scree, loadings, biplot, projection
    figures/concept_pca_*                  explanatory diagrams for the Overview
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"

RANDOM_STATE = 5612

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
        "#008300", "#e87ba4", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK_SOFT, GRID, SURFACE = "#16212e", "#46586c", "#e4e8ee", "#ffffff"
SIGNAL = "#c8102e"
DIV = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#86b6ef", "#f0efec", "#f0a3a2", "#c22d2c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelsize": 10.5, "axes.labelcolor": INK_SOFT,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
})

FEATURES = [
    "n_complaints", "crash_rate", "fire_rate", "pct_severe",
    "injuries_per_100_complaints", "median_days_to_report", "mean_vehicle_age",
    "n_recalls", "InvestigationCount", "OverallRating",
    "OverallFrontCrashRating", "OverallSideCrashRating", "RolloverRating",
    "RolloverPossibility", "has_fcw", "has_ldw",
]

NICE = {
    "log_n_complaints": "Complaint volume", "crash_rate": "Crash rate",
    "fire_rate": "Fire rate", "pct_severe": "Severe share",
    "injuries_per_100_complaints": "Injuries per 100",
    "median_days_to_report": "Days to report", "mean_vehicle_age": "Vehicle age",
    "log_n_recalls": "Recalls", "log_InvestigationCount": "Investigations",
    "OverallRating": "Overall stars", "OverallFrontCrashRating": "Front stars",
    "OverallSideCrashRating": "Side stars", "RolloverRating": "Rollover stars",
    "RolloverPossibility": "Rollover probability",
    "has_fcw": "Collision warning", "has_ldw": "Lane warning",
}


def frame(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"    {name}", flush=True)


# ======================================================================

def prepare():
    df = pd.read_csv(CLEAN / "vehicle_model_year.csv")
    X = df[FEATURES].copy()
    for col in ["n_complaints", "n_recalls", "InvestigationCount"]:
        X[col] = np.log1p(X[col])
        X = X.rename(columns={col: f"log_{col}"})

    # PCA maximises variance, so an unscaled variable with a large numeric
    # range would dominate the first component purely through its units.
    # Standardising makes the analysis run on the correlation matrix.
    Xs = pd.DataFrame(StandardScaler().fit_transform(X),
                      columns=X.columns, index=X.index)
    print(f"  input matrix: {Xs.shape}  (mean {Xs.values.mean():.2e}, "
          f"sd {Xs.values.std():.3f})")
    Xs.round(4).to_csv(CLEAN / "pca_sample.csv", index=False)
    return df, X, Xs


def run_pca(Xs):
    pca = PCA(random_state=RANDOM_STATE).fit(Xs)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    table = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(ev))],
        "eigenvalue": ev.round(4),
        "variance_explained_pct": (evr * 100).round(2),
        "cumulative_pct": (cum * 100).round(2),
    })
    print("\n  eigenvalues and variance explained:")
    print(table.head(10).to_string(index=False))

    n95 = int(np.searchsorted(cum, 0.95) + 1)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    kaiser = int((ev > 1).sum())
    print(f"\n  components needed for 90% variance: {n90}")
    print(f"  components needed for 95% variance: {n95}")
    print(f"  eigenvalues greater than 1 (Kaiser criterion): {kaiser}")
    table.to_csv(CLEAN / "pca_eigenvalues.csv", index=False)
    return pca, ev, evr, cum, n90, n95, kaiser


# ======================================================================

def fig_scree(evr, cum, kaiser, n95):
    """Both series are percentages of the same total, so they share one axis.
    A second y-scale would invent a relationship that is not in the data."""
    k = len(evr)
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x, evr * 100, color=SLOT[0], width=0.62, label="Variance explained by each component")
    ax.plot(x, cum * 100, color=SIGNAL, lw=2, marker="o", ms=5,
            mfc=SIGNAL, mec=SURFACE, mew=1.4, label="Cumulative variance explained")
    ax.axhline(95, color=INK_SOFT, lw=1)
    ax.text(k, 96, "95%", ha="right", fontsize=9.5, color=INK_SOFT)
    ax.axvline(n95, color=INK_SOFT, lw=1)
    ax.annotate(f"{n95} components reach 95%", xy=(n95, 95), xytext=(8, -26),
                textcoords="offset points", fontsize=10, color=INK)
    ax.set_title("Scree plot: how much variance each principal component carries")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Percentage of total variance")
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.legend(loc="center right")
    frame(ax)
    save(fig, "fig23_scree.png")


def fig_eigenvalues(ev, kaiser):
    k = len(ev)
    x = np.arange(1, k + 1)
    colors = [SLOT[0] if v > 1 else "#b9c4d0" for v in ev]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x, ev, color=colors, width=0.62)
    ax.axhline(1, color=SIGNAL, lw=1.6)
    ax.text(k, 1.06, "eigenvalue = 1", ha="right", color=SIGNAL, fontsize=10)
    ax.annotate(f"{kaiser} components carry more variance\nthan a single original variable",
                xy=(kaiser, ev[kaiser - 1]), xytext=(14, 40),
                textcoords="offset points", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1))
    ax.set_title("Eigenvalues of the correlation matrix")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue (variance along that eigenvector)")
    ax.set_xticks(x)
    frame(ax)
    save(fig, "fig24_eigenvalues.png")


def fig_loadings(pca, cols, evr, n=4):
    load = pd.DataFrame(pca.components_[:n].T,
                        index=[NICE.get(c, c) for c in cols],
                        columns=[f"PC{i+1}\n({evr[i]*100:.1f}%)" for i in range(n)])
    order = load["PC1\n(%.1f%%)" % (evr[0] * 100)].sort_values().index
    load = load.loc[order]

    fig, ax = plt.subplots(figsize=(7.4, 7))
    im = ax.imshow(load.values, cmap=DIV, vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(n), load.columns, fontsize=10)
    ax.set_yticks(range(len(load)), load.index, fontsize=10)
    for i in range(load.shape[0]):
        for j in range(load.shape[1]):
            v = load.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.6,
                    color=SURFACE if abs(v) > 0.4 else INK)
    ax.set_title("Eigenvector loadings:\nhow each original variable builds each component",
                 fontsize=12.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Loading", color=INK_SOFT)
    cb.outline.set_visible(False)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig25_loadings.png")


def fig_biplot(pca, scores, cols, ev, evr):
    """Correlation circle.

    Each arrow is one original variable, placed at its correlation with PC1
    and PC2 (loading x sqrt(eigenvalue)). Every arrow therefore sits inside
    the unit circle, which keeps all sixteen variables on the plot instead of
    sending them off the axes as a raw-loading biplot does. Arrows pointing
    the same way are correlated; opposite means inversely related; a right
    angle means unrelated.
    """
    corr = pca.components_[:2].T * np.sqrt(ev[:2])

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.4)
    ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color=GRID, lw=1, ls=(0, (1, 3)))
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)

    # Variables that behave alike point the same way, which is the finding -
    # but it also stacks their labels. Each label is pushed outward along its
    # own arrow until it clears the ones already placed.
    placed = []
    for i, c in enumerate(cols):
        x, y = corr[i]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=SIGNAL,
                                    lw=1.5, alpha=0.9), zorder=3)
        r = np.hypot(x, y) or 1
        ux, uy = x / r, y / r
        pad = 0.075
        while any(np.hypot(x + pad * ux - px, y + pad * uy - py) < 0.17
                  for px, py in placed) and pad < 0.62:
            pad += 0.055
        lx, ly = x + pad * ux, y + pad * uy
        placed.append((lx, ly))
        ax.text(lx, ly, NICE.get(c, c), fontsize=9.5, color=INK, zorder=4,
                ha="left" if x >= 0 else "right",
                va="bottom" if y >= 0 else "top")

    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    ax.set_title("How each variable relates to the first two components")
    ax.set_xlabel(f"Correlation with PC1 ({evr[0]*100:.1f}% of variance)")
    ax.set_ylabel(f"Correlation with PC2 ({evr[1]*100:.1f}% of variance)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    save(fig, "fig26_correlation_circle.png")


def fig_projection(scores, clusters, evr):
    """Vehicles in component space, coloured by the clusters found earlier.
    The clusters were fitted on the full 16-dimensional data, so their
    separation here is evidence that two components retain real structure."""
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    for i, c in enumerate(sorted(clusters.unique())):
        m = (clusters == c).values
        ax.scatter(scores[m, 0], scores[m, 1], s=46,
                   color=SLOT[i % len(SLOT)], marker=MARKERS[i % len(MARKERS)],
                   alpha=0.8, edgecolor=SURFACE, linewidth=0.9,
                   label=f"Cluster {c} (n={m.sum()})")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.set_title("The seven clusters projected onto the first two components")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% of variance)")
    ax.legend(loc="upper right", fontsize=9)
    frame(ax, axis="both")
    save(fig, "fig27_pca_projection_clusters.png")


# ======================================================================
# Concept diagrams for the PCA Overview
# ======================================================================

def concept_pca_projection():
    rng = np.random.default_rng(11)
    n = 160
    t = rng.normal(0, 1.9, n)
    pts = np.column_stack([t + rng.normal(0, 0.42, n),
                           0.62 * t + rng.normal(0, 0.42, n)])
    pts -= pts.mean(0)
    p = PCA(2).fit(pts)
    v1, v2 = p.components_[0], p.components_[1]
    s1, s2 = np.sqrt(p.explained_variance_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))

    ax1.scatter(pts[:, 0], pts[:, 1], s=34, color="#b9c4d0",
                edgecolor=SURFACE, linewidth=0.6)
    for v, s, lab, col in ((v1, s1, "1st eigenvector\n(most variance)", SIGNAL),
                           (v2, s2, "2nd eigenvector", SLOT[0])):
        ax1.annotate("", xy=v * s * 2.3, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
        ax1.text(*(v * s * 2.7), lab, color=col, fontsize=10,
                 ha="center", va="center")
    ax1.set_title("Eigenvectors point along the directions of greatest spread",
                  fontsize=11.5)
    ax1.set_xlabel("Variable A"); ax1.set_ylabel("Variable B")
    ax1.set_aspect("equal")
    frame(ax1, axis="both")

    proj = pts @ v1
    ax2.scatter(proj, np.zeros_like(proj), s=34, color=SIGNAL, alpha=0.5,
                edgecolor="none")
    for i in range(0, n, 3):
        ax2.plot([pts[i] @ v1, pts[i] @ v1], [0, 0.42], color=GRID, lw=0.7, zorder=0)
    ax2.scatter(pts @ v1, (pts @ v2) + 0.9, s=34, color="#b9c4d0",
                edgecolor=SURFACE, linewidth=0.6)
    ax2.set_yticks([0, 0.9], ["After:\n1 dimension", "Before:\n2 dimensions"], fontsize=10)
    ax2.set_xlabel("Position along the 1st principal component")
    ax2.set_title(f"Dropping to one dimension keeps "
                  f"{p.explained_variance_ratio_[0]*100:.0f}% of the variance",
                  fontsize=11.5)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.grid(axis="x")
    ax2.set_axisbelow(True)
    save(fig, "concept_pca_projection.png")


def concept_dimensionality():
    """Why high-dimensional data is hard: distances between points stop
    separating as dimensions grow."""
    rng = np.random.default_rng(3)
    dims = [2, 5, 10, 25, 50, 100, 200, 500]
    ratios = []
    for d in dims:
        pts = rng.normal(size=(220, d))
        dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(pts), 1)
        dd = dist[iu]
        ratios.append(dd.min() / dd.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(dims, ratios, color=SIGNAL, lw=2, marker="o", ms=6,
             mfc=SIGNAL, mec=SURFACE, mew=1.5)
    ax1.set_xscale("log")
    ax1.set_title("Nearest and furthest points converge\nas dimensions increase",
                  fontsize=11.5)
    ax1.set_xlabel("Number of dimensions (log scale)")
    ax1.set_ylabel("Closest distance / furthest distance")
    ax1.set_ylim(0, 1)
    frame(ax1)

    # Sampling density: points needed to keep the same coverage per dimension
    d = np.arange(1, 11)
    need = 10.0 ** d
    ax2.plot(d, need, color=SLOT[0], lw=2, marker="o", ms=6,
             mfc=SLOT[0], mec=SURFACE, mew=1.5)
    ax2.set_yscale("log")
    ax2.set_title("Data needed to keep the same coverage\ngrows exponentially",
                  fontsize=11.5)
    ax2.set_xlabel("Number of dimensions")
    ax2.set_ylabel("Samples required (log scale)")
    ax2.set_xticks(d)
    frame(ax2)
    save(fig, "concept_dimensionality.png")


# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("PCA ANALYSIS")
    print("=" * 72)

    df, X, Xs = prepare()
    pca, ev, evr, cum, n90, n95, kaiser = run_pca(Xs)

    scores = pca.transform(Xs)
    cols = list(Xs.columns)

    pd.DataFrame(pca.components_, columns=cols,
                 index=[f"PC{i+1}" for i in range(len(ev))]
                 ).round(4).to_csv(CLEAN / "pca_components.csv")
    pd.DataFrame(scores[:, :5], columns=[f"PC{i+1}" for i in range(5)]
                 ).round(4).assign(
        make=df["make"], model=df["model"], model_year=df["model_year"],
        segment=df["segment"]).to_csv(CLEAN / "pca_scores.csv", index=False)

    print("\n  top loadings on PC1:")
    pc1 = pd.Series(pca.components_[0], index=cols).sort_values()
    print(pd.concat([pc1.head(4), pc1.tail(4)]).round(3).to_string())
    print("\n  top loadings on PC2:")
    pc2 = pd.Series(pca.components_[1], index=cols).sort_values()
    print(pd.concat([pc2.head(4), pc2.tail(4)]).round(3).to_string())

    print("\n  figures:")
    fig_scree(evr, cum, kaiser, n95)
    fig_eigenvalues(ev, kaiser)
    fig_loadings(pca, cols, evr)
    fig_biplot(pca, scores, cols, ev, evr)

    assign = pd.read_csv(CLEAN / "cluster_assignments.csv")
    fig_projection(scores, assign["kmeans_cluster"], evr)

    concept_pca_projection()
    concept_dimensionality()

    print(f"\n  n90={n90} n95={n95} kaiser={kaiser} "
          f"pc1={evr[0]*100:.1f}% pc2={evr[1]*100:.1f}% pc3={evr[2]*100:.1f}%")
    print("done.")
