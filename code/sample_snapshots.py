"""
sample_snapshots.py  --  CSCI 5612 Project, Module 2 Part 2
Renders small images of the exact matrices fed to clustering and to PCA, so
the Data Prep sections can show the reader what the input actually looks like.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"

INK, INK_SOFT, GRID = "#16212e", "#46586c", "#e4e8ee"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def snapshot(df, title, fname, nrows=6, width=15.5, fs=7.2):
    sub = df.head(nrows).copy()
    for c in sub.columns:
        sub[c] = sub[c].map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    cols = [c.replace("_", " ").replace("log ", "log ") for c in sub.columns]
    cols = [c if len(c) <= 15 else c[:14] + "." for c in cols]

    fig, ax = plt.subplots(figsize=(width, 0.4 * nrows + 1.5))
    ax.axis("off")
    t = ax.table(cellText=sub.values, colLabels=cols, cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.5)
    for (row, _), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#f5f7fa")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK_SOFT)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=16)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    {fname}")


if __name__ == "__main__":
    print("  sample snapshots:")

    clus = pd.read_csv(CLEAN / "clustering_sample.csv")
    snapshot(clus, f"Clustering input: {clus.shape[0]} vehicles x {clus.shape[1]} "
                   "numeric features, no labels of any kind",
             "sample_clustering_input.png")

    pca = pd.read_csv(CLEAN / "pca_sample.csv")
    snapshot(pca, f"PCA input: the same {pca.shape[1]} features standardised to "
                  "mean 0 and standard deviation 1",
             "sample_pca_input.png")

    assign = pd.read_csv(CLEAN / "cluster_assignments.csv")
    snapshot(assign, "Cluster labels joined back to the held-out identifiers "
                     "AFTER fitting, for interpretation only",
             "sample_cluster_assignments.png", width=12)

    eig = pd.read_csv(CLEAN / "pca_eigenvalues.csv").head(8)
    snapshot(eig, "Eigenvalues and variance explained by the leading components",
             "sample_pca_eigenvalues.png", nrows=8, width=9, fs=8.5)
    print("done.")
