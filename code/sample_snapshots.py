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

    # --- BEFORE: the source table, with labels and raw units -------------
    src = pd.read_csv(CLEAN / "vehicle_model_year.csv")
    snapshot(src[["make", "model", "model_year", "segment", "n_complaints",
                  "crash_rate", "median_days_to_report", "n_recalls",
                  "OverallRating", "RolloverPossibility", "rating_class"]],
             f"BEFORE: the Module 1 table, {src.shape[0]} rows x {src.shape[1]} "
             "columns - text labels present, every column on its own scale",
             "sample_before_transform.png", width=14)

    # --- AFTER (clustering): unlabeled, log-corrected, standardised ------
    clus_before = pd.read_csv(CLEAN / "clustering_before_scaling.csv")
    snapshot(clus_before,
             f"AFTER step 1: labels removed and skewed counts logged - "
             f"{clus_before.shape[0]} x {clus_before.shape[1]}, still on mixed scales",
             "sample_clustering_logged.png")

    clus = pd.read_csv(CLEAN / "clustering_sample.csv")
    snapshot(clus, f"AFTER step 2: standardised to mean 0 and sd 1 - the exact "
                   f"{clus.shape[0]} x {clus.shape[1]} matrix both algorithms receive",
             "sample_clustering_input.png")

    # --- AFTER (PCA) -----------------------------------------------------
    pca_before = pd.read_csv(CLEAN / "pca_before_scaling.csv")
    snapshot(pca_before,
             f"BEFORE standardising: {pca_before.shape[1]} numeric features in "
             "their own units, variances wildly unequal",
             "sample_pca_before.png")

    pca = pd.read_csv(CLEAN / "pca_sample.csv")
    snapshot(pca, f"AFTER standardising: the same {pca.shape[1]} features at "
                  "mean 0 and sd 1, so each contributes one unit of variance",
             "sample_pca_input.png")

    assign = pd.read_csv(CLEAN / "cluster_assignments.csv")
    snapshot(assign, "Cluster labels joined back to the held-out identifiers "
                     "AFTER fitting, for interpretation only",
             "sample_cluster_assignments.png", width=12)

    eig = pd.read_csv(CLEAN / "pca_eigenvalues.csv").head(8)
    snapshot(eig, "Eigenvalues and variance explained by the leading components",
             "sample_pca_eigenvalues.png", nrows=8, width=9, fs=8.5)
    print("done.")
