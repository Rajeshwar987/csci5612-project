"""
clustering_analysis.py  --  CSCI 5612 Project, Module 2 Part 2
Unsupervised clustering of the model-year vehicle safety table.

Clustering accepts ONLY unlabeled numeric data, so the categorical and
identifying columns (make, model, segment, rating_class, ...) are removed
before fitting and are kept aside purely to interpret the result afterwards.

Produces:
    data/clean/clustering_sample.csv      the exact unlabeled numeric matrix
    data/clean/cluster_assignments.csv    labels joined back to each vehicle
    figures/fig17..fig22                  silhouette, k comparison, dendrogram,
                                          agreement matrix, cluster profiles
    figures/concept_*                     explanatory diagrams for the Overview
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 5612

# --- shared chart style (matches the EDA figures) ----------------------
# Seven distinct hues, one per cluster at the selected k, so no hue is ever
# reused. Yellow is deliberately omitted: it is the one slot that fails the
# all-pairs separation check against orange, which matters in a scatter.
# Marker shape repeats the same distinction, so colour is never the only cue.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
        "#008300", "#e87ba4", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK_SOFT, GRID, SURFACE = "#16212e", "#46586c", "#e4e8ee", "#ffffff"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#9ec5f4", "#3987e5", "#256abf", "#0d366b"])
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
# 1. DATA PREP  -  unlabeled numeric only
# ======================================================================

# Raw event counts are deliberately excluded: they scale directly with
# n_complaints and would let fleet size dominate every distance computation.
FEATURES = [
    "n_complaints", "crash_rate", "fire_rate", "pct_severe",
    "injuries_per_100_complaints", "median_days_to_report", "mean_vehicle_age",
    "n_recalls", "InvestigationCount", "OverallRating",
    "OverallFrontCrashRating", "OverallSideCrashRating", "RolloverRating",
    "RolloverPossibility", "has_fcw", "has_ldw",
]

LABELS_HELD_OUT = ["make", "model", "model_year", "segment", "top_component",
                   "rollover_risk_band", "rating_class"]


def prepare():
    df = pd.read_csv(CLEAN / "vehicle_model_year.csv")
    print(f"  model-year table: {df.shape}")

    held_out = df[LABELS_HELD_OUT].copy()      # never seen by the algorithms
    X = df[FEATURES].copy()

    # Complaint volume spans three orders of magnitude; without a log it alone
    # drives the geometry. Counts are log1p-transformed before scaling.
    for col in ["n_complaints", "n_recalls", "InvestigationCount"]:
        X[col] = np.log1p(X[col])
        X = X.rename(columns={col: f"log_{col}"})

    # Near-constant columns carry no information for distance-based methods.
    keep = [c for c in X.columns if X[c].std() > 1e-8]
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        print(f"  dropped zero-variance features: {dropped}")
    X = X[keep]

    # Standardising is mandatory here: the features are on wildly different
    # scales (a 0-1 rate against a 5-point star rating against a log count).
    Xs = pd.DataFrame(StandardScaler().fit_transform(X),
                      columns=X.columns, index=X.index)

    print(f"  unlabeled numeric matrix: {Xs.shape}")
    X.to_csv(CLEAN / "clustering_sample.csv", index=False)
    return df, held_out, X, Xs


# ======================================================================
# 2. K-MEANS
# ======================================================================

K_RANGE = range(2, 16)
K_COMPARE = [3, 5, 7]          # the three k values compared side by side


def run_kmeans(Xs):
    print(f"\n  k-means sweep k={K_RANGE.start}..{K_RANGE.stop - 1}")
    rows = []
    models = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        smallest = int(pd.Series(labels).value_counts().min())
        rows.append({"k": k, "silhouette": sil, "inertia": km.inertia_,
                     "smallest_cluster": smallest})
        models[k] = (km, labels)
        print(f"    k={k:2d}  silhouette={sil:.4f}  inertia={km.inertia_:9.1f}"
              f"  smallest cluster={smallest:3d}")
    return pd.DataFrame(rows), models


def choose_k(km_sweep, hc_sweep):
    """Select k on agreement between the two methods, not on the raw maximum.

    The silhouette score here keeps drifting upward as k grows, which is what
    happens when a dataset has no clean separation: the algorithm is simply
    cutting a continuum into ever smaller pieces, and each piece looks tidy.
    Taking the global maximum would therefore just return the edge of whatever
    range was tested. Instead the choice is restricted to interior local
    maxima of the k-means curve, and among those the k that the cosine-based
    hierarchical clustering also scores highest is taken.
    """
    s = km_sweep["silhouette"].values
    ks = km_sweep["k"].values
    local_max = [int(ks[i]) for i in range(1, len(s) - 1)
                 if s[i] > s[i - 1] and s[i] > s[i + 1]]
    print(f"\n  interior local maxima of the k-means silhouette curve: {local_max}")

    hc = hc_sweep.set_index("k")["silhouette"]
    best_k = max(local_max, key=lambda k: hc.get(k, -1)) if local_max else int(ks[s.argmax()])
    print(f"  hierarchical cosine silhouette at those k: "
          f"{ {k: round(float(hc.get(k, float('nan'))), 4) for k in local_max} }")
    print(f"  -> both methods agree on k={best_k}")
    return best_k


# ======================================================================
# 3. HIERARCHICAL CLUSTERING WITH COSINE SIMILARITY
# ======================================================================

def run_hierarchical(Xs):
    """Average-linkage hierarchical clustering on cosine distance.

    Cosine distance is 1 - cosine similarity. Ward linkage is unavailable
    here because it is only defined for squared Euclidean distance, so
    average linkage is used, which is the standard pairing with cosine.
    """
    print("\n  hierarchical clustering, cosine distance, average linkage")
    dist = pdist(Xs.values, metric="cosine")
    Z = linkage(dist, method="average")

    # A dendrogram is normally read by looking for the largest vertical gap
    # between merges. The final merge joining the two halves of the tree is
    # almost always the biggest gap in any dataset, so reporting it would
    # trivially return k=2 regardless of the data; it is excluded here.
    heights = Z[-12:, 2]
    jumps = np.diff(heights[:-1])
    idx = int(np.argmax(jumps))
    gap_k = len(heights) - 1 - idx
    print(f"    merge heights (top 12): {np.round(heights, 3)}")
    print(f"    root merge at {heights[-1]:.3f} excluded as degenerate")
    print(f"    largest remaining gap after {heights[idx]:.3f} -> k={gap_k}")

    rows = []
    for k in K_RANGE:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_score(squareform(dist), labels, metric="precomputed")
        sizes = pd.Series(labels).value_counts()
        rows.append({"k": k, "silhouette": sil,
                     "smallest_cluster": int(sizes.min()),
                     "n_singletons": int((sizes == 1).sum())})
        print(f"    k={k:2d}  cosine silhouette={sil:.4f}  smallest={sizes.min():3d}")
    sweep = pd.DataFrame(rows)

    # A cluster containing one vehicle has a silhouette width of 1 by
    # definition, so any k that strands singletons is rewarded for splitting
    # off outliers rather than for finding structure. Those solutions are
    # excluded before the peak is read off.
    usable = sweep[sweep["n_singletons"] == 0]
    sil_k = int(usable.loc[usable["silhouette"].idxmax(), "k"])
    excluded = sorted(sweep.loc[sweep["n_singletons"] > 0, "k"].tolist())
    print(f"    excluded for producing singleton clusters: k={excluded}")
    print(f"    cosine silhouette peaks at k={sil_k} among usable solutions")
    return Z, dist, sweep, gap_k, sil_k


# ======================================================================
# 4. FIGURES
# ======================================================================

def fig_silhouette_curve(km_sweep, hc_sweep, best_k):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(km_sweep["k"], km_sweep["silhouette"], color=SLOT[0], lw=2,
            marker="o", ms=6, mfc=SLOT[0], mec=SURFACE, mew=1.6, label="k-means (Euclidean)")
    ax.plot(hc_sweep["k"], hc_sweep["silhouette"], color=SLOT[1], lw=2,
            marker="s", ms=6, mfc=SLOT[1], mec=SURFACE, mew=1.6,
            label="Hierarchical (cosine)")
    ax.axvline(best_k, color=INK_SOFT, lw=1)
    ax.annotate(f"best k = {best_k}", xy=(best_k, km_sweep["silhouette"].max()),
                xytext=(8, 6), textcoords="offset points", fontsize=10, color=INK)
    ax.set_title("Silhouette score by number of clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_xticks(list(K_RANGE))
    ax.legend(loc="upper right")
    frame(ax)
    save(fig, "fig17_silhouette_by_k.png")


def fig_silhouette_plot(Xs, labels, k):
    vals = silhouette_samples(Xs, labels)
    avg = vals.mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    y = 0
    for i in range(k):
        v = np.sort(vals[labels == i])
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v,
                         facecolor=SLOT[i % len(SLOT)], edgecolor="none")
        ax.text(-0.045, y + len(v) / 2, f"Cluster {i + 1}", va="center",
                ha="right", fontsize=10, color=INK)
        y += len(v) + 8
    ax.axvline(avg, color=INK, lw=1.4)
    ax.text(avg + 0.01, y * 0.98, f"mean {avg:.3f}", fontsize=10, color=INK, va="top")
    ax.set_title(f"Per-vehicle silhouette widths at k = {k}")
    ax.set_xlabel("Silhouette width")
    ax.set_yticks([])
    ax.set_xlim(min(-0.25, vals.min() - 0.05), max(0.6, vals.max() + 0.05))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    save(fig, "fig18_silhouette_widths.png")


def fig_k_comparison(Xs, models, coords, best_k):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
    for ax, k in zip(axes, K_COMPARE):
        _, labels = models[k]
        sil = silhouette_score(Xs, labels)
        for i in range(k):
            m = labels == i
            ax.scatter(coords[m, 0], coords[m, 1], s=42,
                       color=SLOT[i % len(SLOT)], marker=MARKERS[i % len(MARKERS)],
                       alpha=0.75, edgecolor=SURFACE, linewidth=0.9,
                       label=f"C{i + 1} (n={m.sum()})")
        title = f"k = {k}   silhouette {sil:.3f}"
        if k == best_k:
            title += "   <- best"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Principal component 1")
        ax.legend(loc="upper right", fontsize=8.5, handletextpad=0.2)
        frame(ax, axis="both")
    axes[0].set_ylabel("Principal component 2")
    fig.suptitle("k-means solutions at three values of k, drawn in PCA space",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "fig19_kmeans_k_comparison.png")


def fig_dendrogram(Z, suggested_k, held_out):
    """Full tree. 229 leaf labels cannot be set legibly at any sensible figure
    size, so the leaves are left unlabelled and the branches carry the reading;
    fig20b gives the labelled, truncated view."""
    cut = Z[-(suggested_k - 1), 2] if suggested_k > 1 else Z[-1, 2]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=cut,
               above_threshold_color=INK_SOFT)
    ax.axhline(cut, color="#c8102e", lw=1.4, zorder=0)
    ax.text(ax.get_xlim()[1], cut + 0.012,
            f"cut at {cut:.3f}  ->  k = {suggested_k} ",
            color="#c8102e", fontsize=10.5, va="bottom", ha="right")
    ax.set_title("Hierarchical clustering of 229 vehicles, "
                 "cosine distance with average linkage")
    ax.set_ylabel("Cosine distance at merge")
    ax.set_xlabel("Individual vehicles (229 leaves)")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    save(fig, "fig20_dendrogram.png")

    # Truncated companion: the last 20 merges, each leaf annotated with how
    # many vehicles it stands for.
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=20, show_contracted=True,
               color_threshold=cut, above_threshold_color=INK_SOFT,
               leaf_font_size=10)
    ax.axhline(cut, color="#c8102e", lw=1.4, zorder=0)
    ax.set_title(f"The same tree truncated to its final 20 merges, cut at k = {suggested_k}")
    ax.set_ylabel("Cosine distance at merge")
    ax.set_xlabel("Branch, with the number of vehicles it contains in brackets")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    save(fig, "fig20b_dendrogram_truncated.png")


def fig_agreement(km_labels, hc_labels, ari):
    ct = pd.crosstab(pd.Series(km_labels, name="k-means cluster") + 1,
                     pd.Series(hc_labels, name="Hierarchical cluster"))
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(ct.values, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(ct.shape[1]), [f"H{c}" for c in ct.columns])
    ax.set_yticks(range(ct.shape[0]), [f"K{r}" for r in ct.index])
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=11,
                    color=SURFACE if v > ct.values.max() * 0.55 else INK)
    ax.set_title(f"Where the two methods agree\nadjusted Rand index = {ari:.3f}",
                 fontsize=12.5)
    ax.set_xlabel("Hierarchical (cosine) cluster")
    ax.set_ylabel("k-means cluster")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig21_kmeans_vs_hclust.png")


def fig_profiles(Xs, labels, k):
    prof = Xs.copy()
    prof["cluster"] = labels
    means = prof.groupby("cluster").mean()
    means.index = [f"Cluster {i + 1}\n(n={(labels == i).sum()})" for i in means.index]
    nice = [c.replace("log_", "log ").replace("_", " ") for c in means.columns]

    fig, ax = plt.subplots(figsize=(12, 3.6))
    im = ax.imshow(means.values, cmap=DIV, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(nice)), nice, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(means)), means.index, fontsize=9.5)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            v = means.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.8,
                    color=SURFACE if abs(v) > 1.2 else INK)
    ax.set_title(f"What distinguishes each cluster at k = {k} "
                 "(standard deviations from the overall mean)", fontsize=12.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.012)
    cb.set_label("z-score", color=INK_SOFT)
    cb.outline.set_visible(False)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig22_cluster_profiles.png")


# ======================================================================
# 5. CONCEPT DIAGRAMS for the Overview section
# ======================================================================

def concept_partitional_vs_hierarchical():
    rng = np.random.default_rng(7)
    a = rng.normal([0, 0], 0.55, (26, 2))
    b = rng.normal([2.6, 0.6], 0.55, (26, 2))
    c = rng.normal([1.3, 2.7], 0.55, (26, 2))
    pts = np.vstack([a, b, c])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    km = KMeans(3, n_init=10, random_state=1).fit(pts)
    for i in range(3):
        m = km.labels_ == i
        ax1.scatter(pts[m, 0], pts[m, 1], s=46, color=SLOT[i],
                    marker=MARKERS[i], edgecolor=SURFACE, linewidth=0.8)
    ax1.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=260, marker="X", color=INK, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax1.set_title("Partitional (k-means)\nk is chosen first; every point joins exactly one group",
                  fontsize=11.5)
    ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_color(GRID)

    Z = linkage(pts, method="average", metric="cosine")
    dendrogram(Z, ax=ax2, no_labels=True, color_threshold=Z[-2, 2],
               above_threshold_color=INK_SOFT)
    ax2.set_title("Hierarchical (agglomerative)\nevery merge is recorded; k is chosen after, by cutting",
                  fontsize=11.5)
    ax2.set_ylabel("Distance at merge")
    ax2.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["left"].set_color(GRID)
    save(fig, "concept_clustering_types.png")


def concept_distance_metrics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    p, q = np.array([3.0, 1.0]), np.array([1.4, 2.6])

    for ax in (ax1, ax2):
        ax.set_xlim(-0.4, 4.2); ax.set_ylim(-0.4, 3.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.plot([0, 0], [-0.4, 3.4], color=GRID, lw=1)
        ax.plot([-0.4, 4.2], [0, 0], color=GRID, lw=1)

    ax1.plot([p[0], q[0]], [p[1], q[1]], color=SLOT[1], lw=2.4, zorder=1)
    ax1.text((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2 + 0.18,
             "straight-line\ndistance", color=SLOT[1], fontsize=10.5)
    ax1.set_title("Euclidean distance\nhow far apart the points are", fontsize=11.5)

    for v, lab in ((p, "A"), (q, "B")):
        ax2.annotate("", xy=v, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=SLOT[0], lw=2.2))
        ax2.text(v[0] + 0.1, v[1] + 0.1, lab, fontsize=12, color=INK, fontweight="bold")
    ang = np.linspace(np.arctan2(p[1], p[0]), np.arctan2(q[1], q[0]), 60)
    ax2.plot(0.85 * np.cos(ang), 0.85 * np.sin(ang), color=SLOT[3], lw=2)
    ax2.text(1.05, 1.05, "angle between\nthe directions", color=SLOT[3], fontsize=10.5)
    ax2.set_title("Cosine distance\nhow different the directions are, ignoring magnitude",
                  fontsize=11.5)

    for ax, v, lab in ((ax1, p, "A"), (ax1, q, "B")):
        ax.scatter(*v, s=90, color=INK, zorder=3)
        ax.text(v[0] + 0.1, v[1] + 0.12, lab, fontsize=12, color=INK, fontweight="bold")
    save(fig, "concept_distance_metrics.png")


# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("CLUSTERING ANALYSIS")
    print("=" * 72)

    df, held_out, X, Xs = prepare()

    km_sweep, models = run_kmeans(Xs)
    Z, dist, hc_sweep, gap_k, sil_k = run_hierarchical(Xs)
    best_k = choose_k(km_sweep, hc_sweep)
    suggested_k = sil_k

    hc_labels = fcluster(Z, t=suggested_k, criterion="maxclust")
    km_best_labels = models[best_k][1]
    ari = adjusted_rand_score(km_best_labels, hc_labels)
    print(f"\n  adjusted Rand index (k-means k={best_k} vs hclust k={suggested_k}): {ari:.3f}")

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)

    print("\n  figures:")
    fig_silhouette_curve(km_sweep, hc_sweep, best_k)
    fig_silhouette_plot(Xs, km_best_labels, best_k)
    fig_k_comparison(Xs, models, coords, best_k)
    fig_dendrogram(Z, suggested_k, held_out)
    fig_agreement(km_best_labels, hc_labels, ari)
    fig_profiles(Xs, km_best_labels, best_k)
    concept_partitional_vs_hierarchical()
    concept_distance_metrics()

    out = held_out.copy()
    out["kmeans_cluster"] = km_best_labels + 1
    out["hclust_cluster"] = hc_labels
    out.to_csv(CLEAN / "cluster_assignments.csv", index=False)

    km_sweep.to_csv(CLEAN / "kmeans_silhouette_sweep.csv", index=False)

    print("\n  cluster composition by segment (k-means):")
    print(pd.crosstab(out["kmeans_cluster"], out["segment"]).to_string())
    print("\n  cluster means on key measures:")
    prof = df[FEATURES].copy()
    prof["cluster"] = km_best_labels + 1
    print(prof.groupby("cluster").mean().round(3).to_string())
    print(f"\n  best_k={best_k}  suggested_k={suggested_k}  ARI={ari:.3f}")
    print("done.")
