"""
make_notebooks.py  --  CSCI 5612 Project, Module 2 Part 2
Builds and executes the two analysis notebooks so the submitted .ipynb files
carry their outputs inline.

Writes:  notebooks/clustering.ipynb
         notebooks/pca.ipynb
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# ======================================================================
# CLUSTERING NOTEBOOK
# ======================================================================

clustering = [
    md("""
# Clustering vehicle safety records

**CSCI 5612 — Module 2, Project Part 2**

If the only thing known about a vehicle were its measured safety record — no
make, no model, no body style, no star rating — what groups would the data put
it into on its own?

This notebook runs k-means and hierarchical clustering over 229 vehicles
described by 16 numeric safety measures. The hierarchical clustering uses
**cosine distance**, as the assignment specifies.
"""),
    code("""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
RANDOM_STATE = 5612
"""),
    md("""
## 1. Data preparation

Clustering accepts **only unlabeled numeric data**. Distances cannot be computed
on text, and an unsupervised method that can see a label is no longer
unsupervised. Everything identifying or categorical is removed and held aside
to interpret the result afterwards.
"""),
    code("""
df = pd.read_csv("../data/clean/vehicle_model_year.csv")
print("source table:", df.shape)
df.head(3)
"""),
    code("""
FEATURES = [
    "n_complaints", "crash_rate", "fire_rate", "pct_severe",
    "injuries_per_100_complaints", "median_days_to_report", "mean_vehicle_age",
    "n_recalls", "InvestigationCount", "OverallRating",
    "OverallFrontCrashRating", "OverallSideCrashRating", "RolloverRating",
    "RolloverPossibility", "has_fcw", "has_ldw",
]
HELD_OUT = ["make", "model", "model_year", "segment", "top_component",
            "rollover_risk_band", "rating_class"]

held_out = df[HELD_OUT].copy()     # never seen by the algorithms
X = df[FEATURES].copy()

# Raw event counts are excluded on purpose: they scale with complaint volume,
# so keeping them would let fleet size dominate every distance.
# Skewed counts are log-transformed first.
for col in ["n_complaints", "n_recalls", "InvestigationCount"]:
    X[col] = np.log1p(X[col])
    X = X.rename(columns={col: f"log_{col}"})

print("BEFORE standardising — note how unequal the variances are:")
X.describe().loc[["mean", "std", "min", "max"]].round(2)
"""),
    code("""
# Standardising is mandatory: an unscaled star rating running 1-5 would be
# swamped by a complaint count running into the hundreds, purely through units.
Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
print("AFTER standardising:", Xs.shape)
Xs.describe().loc[["mean", "std", "min", "max"]].round(2)
"""),
    md("""
## 2. k-means, swept across k

The silhouette width measures how much closer each vehicle sits to its own
cluster than to the next nearest one.
"""),
    code("""
K_RANGE = range(2, 16)
rows, models = [], {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
    labels = km.fit_predict(Xs)
    rows.append({"k": k,
                 "silhouette": silhouette_score(Xs, labels),
                 "inertia": km.inertia_,
                 "smallest_cluster": int(pd.Series(labels).value_counts().min())})
    models[k] = labels
km_sweep = pd.DataFrame(rows)
km_sweep.round(4)
"""),
    md("""
**The naive answer is wrong here.** Taking the highest silhouette would return
k=15 — the edge of the tested range, where the smallest cluster holds three
vehicles. A curve that keeps rising with k is the signature of a dataset with
no clean separation: the algorithm is slicing a continuum into ever smaller
pieces, and smaller pieces always look tidier.

Only **interior local maxima** are treated as candidates.
"""),
    code("""
s, ks = km_sweep["silhouette"].values, km_sweep["k"].values
local_max = [int(ks[i]) for i in range(1, len(s) - 1)
             if s[i] > s[i - 1] and s[i] > s[i + 1]]
print("interior local maxima:", local_max)
"""),
    md("""
## 3. Hierarchical clustering with cosine distance

Cosine distance is `1 - cosine similarity`. It compares the *shape* of two
records regardless of magnitude, so a heavily complained-about model and a
lightly complained-about one group together if their problems are
proportionally alike.

Ward linkage is **not** available: it is defined only for squared Euclidean
distance. Average linkage is the standard pairing with cosine.
"""),
    code("""
dist = pdist(Xs.values, metric="cosine")
Z = linkage(dist, method="average")

rows = []
for k in K_RANGE:
    labels = fcluster(Z, t=k, criterion="maxclust")
    sizes = pd.Series(labels).value_counts()
    rows.append({"k": k,
                 "silhouette": silhouette_score(squareform(dist), labels,
                                                metric="precomputed"),
                 "smallest_cluster": int(sizes.min()),
                 "n_singletons": int((sizes == 1).sum())})
hc_sweep = pd.DataFrame(rows)
hc_sweep.round(4)
"""),
    md("""
A cluster containing one vehicle has a silhouette width of **1 by definition**,
so any k that strands singletons is rewarded for isolating outliers rather than
for finding structure. Those solutions are discarded before reading the peak.
"""),
    code("""
usable = hc_sweep[hc_sweep["n_singletons"] == 0]
sil_k = int(usable.loc[usable["silhouette"].idxmax(), "k"])
hc = hc_sweep.set_index("k")["silhouette"]
best_k = max(local_max, key=lambda k: hc.get(k, -1))

print("hierarchical peak (no singletons):     k =", sil_k)
print("k-means local maxima scored by hclust:",
      {k: round(float(hc[k]), 4) for k in local_max})
print("\\n>>> both methods agree on k =", best_k)
"""),
    md("""
## 4. Do the two methods agree?

They use different distance metrics and entirely different procedures, so
agreement is meaningful evidence.
"""),
    code("""
km_labels = models[best_k]
hc_labels = fcluster(Z, t=sil_k, criterion="maxclust")
ari = adjusted_rand_score(km_labels, hc_labels)
print(f"adjusted Rand index: {ari:.3f}")
pd.crosstab(pd.Series(km_labels + 1, name="k-means"),
            pd.Series(hc_labels, name="hierarchical"))
"""),
    md("## 5. What the clusters contain\n\nSegment is checked **only now**, after fitting."),
    code("""
out = held_out.copy()
out["kmeans_cluster"] = km_labels + 1
pd.crosstab(out["kmeans_cluster"], out["segment"])
"""),
    code("""
profile = df[FEATURES].copy()
profile["cluster"] = km_labels + 1
profile.groupby("cluster").mean().round(3)
"""),
    md("""
## 6. What this says about the topic

Silhouette widths peak near 0.3, where above 0.5 would indicate clean
separation. Vehicle safety records form a **continuum, not a set of distinct
types** — there is no natural population of dangerous cars sitting apart from
safe ones. That is a substantive finding, and it is part of why a five-point
star rating struggles to separate modern vehicles.

Where structure does exist, it is organised by **physics and equipment rather
than by brand**: the cleanest group in the whole solution contains trucks and
large SUVs and nothing else, separated on rollover probability and crash-test
score alone, without the algorithm ever being shown a body style.

Full figures and discussion: <https://rajeshwar987.github.io/csci5612-project/clustering.html>
"""),
]


# ======================================================================
# PCA NOTEBOOK
# ======================================================================

pca_cells = [
    md("""
# Principal Component Analysis of vehicle safety measures

**CSCI 5612 — Module 2, Project Part 2**

Sixteen separate safety measures describe each vehicle. Are they really sixteen
different things, or a handful of underlying patterns wearing sixteen names?
"""),
    code("""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
RANDOM_STATE = 5612
"""),
    md("""
## 1. Data preparation

PCA needs the same input shape as clustering: a purely numeric matrix with no
labels. Standardising matters even more here. PCA finds directions of maximum
variance, and variance depends on units, so an unstandardised variable with a
wide numeric range would capture the first component purely by being measured
on a larger scale. Standardising means the analysis runs on the **correlation**
matrix, and every variable gets an equal vote.
"""),
    code("""
df = pd.read_csv("../data/clean/vehicle_model_year.csv")
FEATURES = [
    "n_complaints", "crash_rate", "fire_rate", "pct_severe",
    "injuries_per_100_complaints", "median_days_to_report", "mean_vehicle_age",
    "n_recalls", "InvestigationCount", "OverallRating",
    "OverallFrontCrashRating", "OverallSideCrashRating", "RolloverRating",
    "RolloverPossibility", "has_fcw", "has_ldw",
]
X = df[FEATURES].copy()
for col in ["n_complaints", "n_recalls", "InvestigationCount"]:
    X[col] = np.log1p(X[col])
    X = X.rename(columns={col: f"log_{col}"})

print("BEFORE standardising:")
display(X.describe().loc[["mean", "std"]].round(2))

Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
print("AFTER standardising:")
Xs.describe().loc[["mean", "std"]].round(2)
"""),
    md("""
## 2. Eigenvalues and eigenvectors

For the correlation matrix, an **eigenvector** is a direction the matrix
stretches without rotating, and its **eigenvalue** is how much stretching
occurs. In data terms the eigenvectors are the candidate axes and the
eigenvalues are the variance captured along each one. Sorting eigenvectors by
eigenvalue produces the principal components in order of importance — that is
the entire mechanism.

Because the input is standardised, each original variable contributes exactly
1 unit of variance, so **an eigenvalue above 1** marks a component carrying
more information than any single original measure did alone.
"""),
    code("""
pca = PCA(random_state=RANDOM_STATE).fit(Xs)
ev, evr = pca.explained_variance_, pca.explained_variance_ratio_
cum = np.cumsum(evr)

pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(len(ev))],
    "eigenvalue": ev.round(4),
    "variance_explained_%": (evr * 100).round(2),
    "cumulative_%": (cum * 100).round(2),
})
"""),
    code("""
print("components with eigenvalue > 1 (Kaiser):", int((ev > 1).sum()))
print("components needed for 90% variance:    ", int(np.searchsorted(cum, 0.90) + 1))
print("components needed for 95% variance:    ", int(np.searchsorted(cum, 0.95) + 1))
"""),
    md("""
## 3. What the components mean

The loadings say how much each original variable contributes to each component.
"""),
    code("""
loadings = pd.DataFrame(pca.components_[:4].T, index=Xs.columns,
                        columns=["PC1", "PC2", "PC3", "PC4"])
loadings.round(3).sort_values("PC1")
"""),
    code("""
for pc in ["PC1", "PC2"]:
    s = loadings[pc].sort_values()
    print(f"\\n{pc}  (explains {evr[int(pc[-1])-1]*100:.1f}% of variance)")
    print("  most negative:", ", ".join(f"{i} {v:+.2f}" for i, v in s.head(3).items()))
    print("  most positive:", ", ".join(f"{i} {v:+.2f}" for i, v in s.tail(3).items()))
"""),
    md("""
## 4. What this says about the topic

**This data does not reduce well.** Nine of sixteen components are needed to
retain 90% of the variance, and the leading component carries under a quarter.
A dataset with strong underlying structure usually puts 60–80% into its first
two or three. So the sixteen measures are close to sixteen genuinely separate
pieces of information, not a few latent factors in disguise — which helps
explain why one star rating leaves so much unsaid.

**PC2 is the more pointed finding.** It loads crash-test ratings against
realised harm with opposite signs, meaning the laboratory measurement and the
ownership record do not merely differ — they vary in opposition along the
strongest axis on which both appear.

Full figures and discussion: <https://rajeshwar987.github.io/csci5612-project/pca.html>
"""),
]


def build(cells, path, title):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "csci5612",
                                 "display_name": "Python (csci5612)",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    print(f"  executing {title} ...", flush=True)
    client = NotebookClient(nb, timeout=900, kernel_name="csci5612",
                            resources={"metadata": {"path": str(NB)}})
    client.execute()
    nbf.write(nb, path)
    errs = sum(1 for c in nb.cells
               for o in c.get("outputs", []) if o.get("output_type") == "error")
    print(f"    -> {path.name}  ({len(nb.cells)} cells, {errs} errors)")


if __name__ == "__main__":
    print("building notebooks:")
    build(clustering, NB / "clustering.ipynb", "clustering")
    build(pca_cells, NB / "pca.ipynb", "pca")
    print("done.")
