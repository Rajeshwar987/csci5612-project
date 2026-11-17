"""
adaboost_analysis.py  --  CSCI 5612 Project, Module 4 Part 4
AdaBoost with decision stumps.

Task: the same label as the Naive Bayes, Decision Tree and logistic regression
work - did this complaint involve a crash - predicted from the structured
attributes of the vehicle and the report. The feature matrix and the 75/25
train/test split are rebuilt exactly as on the Decision Trees tab (verified
against its published samples), so AdaBoost and the Part 3 tree are scored on
the very same 20,154 test complaints.

Settings (number of rounds and learning rate) are chosen on a validation set
carved out of the training set. The test set is used once, to score the
fixed model.

Outputs:
    data/clean/ada_*.csv      labelled sample, split samples, validation
                              curves, chosen settings, stump table, metrics
    figures/ada_*.png         concepts, data images, results
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import warnings

from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)

# scikit-learn warns about feature names inside staged_* iteration even though
# every call passes the same DataFrame columns; the warning is spurious here.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 5612
TEST_SIZE = 0.25
VAL_SIZE = 0.20            # share of the TRAINING set held out for tuning
MAX_ROUNDS = 400
LEARNING_RATES = [0.1, 0.25, 0.5, 1.0]
WEIGHTINGS = ["balanced", "uniform"]     # starting row weights in round 1

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#008300", "#e87ba4", "#e34948"]
INK, INK_SOFT, GRID, SURFACE = "#16212e", "#46586c", "#e4e8ee", "#ffffff"
SIGNAL = "#c8102e"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#9ec5f4", "#3987e5", "#256abf", "#0d366b"])

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


def _text_pts(s, fs, bold=False):
    """Rendered width of a string in points, measured from the font itself."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return TextPath((0, 0), s, size=fs, prop=fp).get_extents().width if s.strip() else 0.0


def table_png(df, title, fname, nrows=6, width=None, fs=8.4):
    """Small image of the first rows of a table. Every column is exactly as
    wide as its widest rendered entry and the image is sized to the table
    itself, so there is no empty space inside cells or around the table."""
    sub = df.head(nrows).copy()
    for c in sub.columns:
        sub[c] = sub[c].map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    labels = [c.replace("_", " ") for c in sub.columns]
    text_cols = set()
    pad = 16.0
    col_pts = np.array([max(_text_pts(l, fs, bold=True), *(_text_pts(v, fs) for v in sub[c])) + pad
                        for l, c in zip(labels, sub.columns)])
    widths = col_pts / col_pts.sum()

    fig = plt.figure(figsize=(col_pts.sum() / 72, 0.3 * (nrows + 1)))
    ax = fig.add_axes([0, 0, 1, 1])        # table fills the canvas exactly
    ax.axis("off")
    t = ax.table(cellText=sub.values, colLabels=labels, colWidths=widths,
                 cellLoc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    for (r, j), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        if j in text_cols:
            cell.get_text().set_horizontalalignment("left")
            cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#f5f7fa")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK_SOFT)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=8)
    # No tight_layout here: it would move the axes and squeeze the measured columns.
    fig.savefig(FIG / fname, dpi=150, bbox_inches="tight", pad_inches=0.08, facecolor=SURFACE)
    plt.close(fig)
    print(f"    {fname}", flush=True)


def metrics(name, y, p):
    return {"model": name,
            "accuracy": accuracy_score(y, p),
            "precision_crash": precision_score(y, p, pos_label="Crash", zero_division=0),
            "recall_crash": recall_score(y, p, pos_label="Crash", zero_division=0),
            "f1_crash": f1_score(y, p, pos_label="Crash", zero_division=0)}


def stumps(n, lr):
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                              n_estimators=n, learning_rate=lr,
                              random_state=RANDOM_STATE)


# ======================================================================
# 1. DATA  (identical to decision_tree_analysis.build_features)
# ======================================================================

def build_features():
    print("\n[1] LABELLED DATA (same construction as the Decision Trees tab)")
    c = pd.read_csv(CLEAN / "complaints_clean.csv.gz", low_memory=False)
    my = pd.read_csv(CLEAN / "vehicle_model_year.csv")
    cols = ["make", "model", "model_year", "OverallRating", "RolloverPossibility",
            "has_fcw", "has_ldw", "n_recalls"]
    c = c.merge(my[cols], on=["make", "model", "model_year"], how="inner")
    c["label"] = np.where(c["crash"], "Crash", "No crash")
    top = c["component_primary"].value_counts().head(12).index
    c["component"] = np.where(c["component_primary"].isin(top),
                              c["component_primary"], "OTHER")

    base = pd.DataFrame({
        "vehicle_age": c["vehicle_age_at_incident"],
        "days_to_report": c["days_to_report"],
        "model_year": c["model_year"],
        "overall_rating": c["OverallRating"],
        "rollover_prob": c["RolloverPossibility"],
        "has_fcw": c["has_fcw"],
        "has_ldw": c["has_ldw"],
        "n_recalls": c["n_recalls"],
        "summary_words": c["summary"].fillna("").astype(str).str.split().str.len(),
    })
    # BEFORE encoding: two text columns that AdaBoost cannot accept.
    before = pd.concat([base, c[["segment", "component"]].reset_index(drop=True)
                        .set_axis(base.index)], axis=1)
    before["label"] = c["label"].values

    # One-hot encoding: each category becomes its own 0/1 column.
    feat = pd.concat([base,
                      pd.get_dummies(c["segment"], prefix="seg").astype(int),
                      pd.get_dummies(c["component"], prefix="cmp").astype(int)],
                     axis=1)
    feat["label"] = c["label"].values
    keep = feat.notna().all(axis=1)
    feat = feat[keep].reset_index(drop=True)
    before = before[keep.values].reset_index(drop=True)
    print(f"    {feat.shape[0]:,} complaints, {before.shape[1]-1} raw columns "
          f"-> {feat.shape[1]-1} numeric features after one-hot encoding")
    before.head(500).to_csv(CLEAN / "ada_labeled_before_encoding.csv", index=False)
    feat.head(500).to_csv(CLEAN / "ada_labeled_encoded.csv", index=False)
    return before, feat


def split(feat):
    print("\n[2] TRAIN / VALIDATION / TEST")
    X, y = feat.drop(columns=["label"]), feat["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=RANDOM_STATE, stratify=y)
    assert not (set(Xtr.index) & set(Xte.index)), "train and test overlap"
    dt_te = pd.read_csv(CLEAN / "dt_test_sample.csv")
    assert np.allclose(Xte.head(len(dt_te)).values, dt_te.drop(columns="label").values), \
        "test set differs from the Decision Trees tab"
    print(f"    train {len(Xtr):,}   test {len(Xte):,}   (same split as the Decision Trees tab)")

    # The validation set is carved out of TRAINING data only.
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=VAL_SIZE,
                                              random_state=RANDOM_STATE, stratify=ytr)
    assert not (set(Xfit.index) & set(Xval.index))
    print(f"    training set split again for tuning: fit {len(Xfit):,}   "
          f"validation {len(Xval):,}")
    Xtr.assign(label=ytr).head(400).to_csv(CLEAN / "ada_train_sample.csv", index=False)
    Xte.assign(label=yte).head(400).to_csv(CLEAN / "ada_test_sample.csv", index=False)
    return Xtr, Xte, ytr, yte, Xfit, Xval, yfit, yval


# ======================================================================
# 2. TUNING ON THE VALIDATION SET
# ======================================================================

def crash_score(model, X):
    """Signed AdaBoost score oriented so that higher means 'more crash-like'."""
    s = model.decision_function(X)
    return s if list(model.classes_).index("Crash") == 1 else -s


def tune(Xfit, yfit, Xval, yval):
    """Settings chosen on the VALIDATION set only.

    For every combination of starting weights and learning rate, MAX_ROUNDS
    stumps are fitted once and the validation score is read off after every
    round with staged_decision_function, which gives the whole 'performance as
    rounds are added' curve from a single fit.

    Selection uses validation average precision (area under the
    precision-recall curve). It judges how well the model ranks crashes above
    non-crashes without depending on any cut-off, which matters here because
    accuracy at the default cut-off on a 4.7% class mostly measures how often
    the model says 'no crash'. Accuracy is recorded alongside for reference.
    """
    print("\n[3] TUNING ON VALIDATION DATA")
    yv = (yval == "Crash").astype(int).values
    pos_cls = None
    rows = []
    for wname in WEIGHTINGS:
        w = compute_sample_weight("balanced", yfit) if wname == "balanced" else None
        for lr in LEARNING_RATES:
            m = stumps(MAX_ROUNDS, lr).fit(Xfit, yfit, sample_weight=w)
            pos_cls = list(m.classes_).index("Crash")
            for k, s in enumerate(m.staged_decision_function(Xval), start=1):
                sc = s if pos_cls == 1 else -s
                pred = sc > 0                                    # default cut-off
                rows.append({"start_weights": wname, "learning_rate": lr, "n_estimators": k,
                             "val_accuracy": float((pred == yv).mean()),
                             "val_f1_default": f1_score(yv, pred, zero_division=0),
                             "val_avg_precision": average_precision_score(yv, sc)})
            g = pd.DataFrame(rows[-MAX_ROUNDS:])
            b = g.loc[g["val_avg_precision"].idxmax()]
            print(f"    weights={wname:<8} lr={lr:<5} best val AP {b['val_avg_precision']:.4f} "
                  f"at {int(b['n_estimators'])} rounds | after 1 round {g['val_avg_precision'].iloc[0]:.4f}"
                  f" | val acc after {MAX_ROUNDS}: {g['val_accuracy'].iloc[-1]:.4f}", flush=True)
    curves = pd.DataFrame(rows)

    # Highest validation AP; among near-ties (within 0.001) prefer fewer rounds.
    top = curves["val_avg_precision"].max()
    best = (curves[curves["val_avg_precision"] >= top - 0.001]
            .sort_values(["n_estimators", "val_avg_precision"], ascending=[True, False])
            .iloc[0])
    print(f"    -> chosen: start weights={best['start_weights']}, "
          f"learning_rate={best['learning_rate']}, n_estimators={int(best['n_estimators'])} "
          f"(val AP {best['val_avg_precision']:.4f}; best was {top:.4f})")

    # Third setting: the decision threshold, also chosen on validation.
    w = compute_sample_weight("balanced", yfit) if best["start_weights"] == "balanced" else None
    m = stumps(int(best["n_estimators"]), float(best["learning_rate"])).fit(Xfit, yfit, sample_weight=w)
    sc = crash_score(m, Xval)
    p, r, t = precision_recall_curve(yv, sc)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f))
    thr = float(t[i])
    print(f"    -> threshold chosen on validation: score > {thr:.4f} "
          f"(val F1 {f[i]:.4f}, precision {p[i]:.3f}, recall {r[i]:.3f}; "
          f"default cut-off 0 gives F1 {f1_score(yv, sc > 0):.4f})")
    thr_curve = pd.DataFrame({"threshold": t, "precision": p[:-1], "recall": r[:-1], "f1": f})

    chosen = {"start_weights": best["start_weights"], "learning_rate": float(best["learning_rate"]),
              "n_estimators": int(best["n_estimators"]), "threshold": round(thr, 5),
              "val_avg_precision": round(float(best["val_avg_precision"]), 5),
              "val_f1_at_threshold": round(float(f[i]), 5),
              "val_accuracy_default_cutoff": round(float(best["val_accuracy"]), 5)}
    curves.round(5).to_csv(CLEAN / "ada_validation_curves.csv", index=False)
    pd.DataFrame([chosen]).to_csv(CLEAN / "ada_chosen_settings.csv", index=False)
    return curves, chosen, thr_curve


# ======================================================================
# 3. FIGURES
# ======================================================================

def fig_curves(curves, chosen):
    cw = chosen["start_weights"]
    other = [w for w in WEIGHTINGS if w != cw][0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for i, (lr, g) in enumerate(curves[curves["start_weights"] == cw].groupby("learning_rate")):
        a1.plot(g["n_estimators"], g["val_accuracy"] * 100, color=SLOT[i], lw=1.8,
                label=f"learning rate {lr:g}")
        a2.plot(g["n_estimators"], g["val_avg_precision"] * 100, color=SLOT[i], lw=1.8,
                label=f"learning rate {lr:g}")
    o = curves[(curves["start_weights"] == other) &
               (curves["learning_rate"] == chosen["learning_rate"])]
    for ax, col in ((a1, "val_accuracy"), (a2, "val_avg_precision")):
        ax.plot(o["n_estimators"], o[col] * 100, color=INK_SOFT, lw=1.4, ls=(0, (2, 2)),
                label=f"{other} starting weights, lr {chosen['learning_rate']:g}")
        ax.axvline(chosen["n_estimators"], color=INK_SOFT, lw=1)
        ax.set_xlabel("Boosting rounds (number of stumps)")
        frame(ax)
    a2.scatter([chosen["n_estimators"]], [chosen["val_avg_precision"] * 100], s=70,
               color=SIGNAL, zorder=5, edgecolor=SURFACE, linewidth=1.5)
    a2.annotate(f"chosen: {chosen['n_estimators']} rounds, lr {chosen['learning_rate']:g}",
                xy=(chosen["n_estimators"], chosen["val_avg_precision"] * 100),
                xytext=(40, -90), textcoords="offset points", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1))
    a1.set_ylabel("Validation accuracy at the default cut-off (%)")
    a1.set_title("Validation accuracy as rounds are added", fontsize=11.5)
    a2.set_ylabel("Validation average precision (%)")
    a2.set_title("Validation average precision as rounds are added", fontsize=11.5)
    a1.legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Tuning AdaBoost on the validation set ({cw} starting weights)",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    save(fig, "ada_fig_validation_curves.png")


def fig_threshold(thr_curve, chosen):
    tc = thr_curve
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(tc["threshold"], tc["precision"] * 100, color=SLOT[0], lw=2, label="Precision (Crash)")
    ax.plot(tc["threshold"], tc["recall"] * 100, color=SLOT[1], lw=2, label="Recall (Crash)")
    ax.plot(tc["threshold"], tc["f1"] * 100, color=SIGNAL, lw=2.4, label="F1 (Crash)")
    ax.axvline(chosen["threshold"], color=INK_SOFT, lw=1)
    ax.axvline(0, color=GRID, lw=1.2)
    ax.annotate(f"chosen cut-off {chosen['threshold']:.3f}\nvalidation F1 "
                f"{chosen['val_f1_at_threshold']:.3f}",
                xy=(chosen["threshold"], chosen["val_f1_at_threshold"] * 100),
                xytext=(-330, 55), textcoords="offset points", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1))
    ax.text(0.01, 96, "default\ncut-off 0", fontsize=9, color=INK_SOFT, va="top")
    ax.set_xlabel("Cut-off on the AdaBoost score (score above it = predict Crash)")
    ax.set_ylabel("Percent, validation set")
    ax.set_title("Choosing the decision cut-off on validation data")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    frame(ax)
    save(fig, "ada_fig_threshold.png")


def fig_grid(curves, chosen):
    checkpoints = [1, 10, 25, 50, 100, 200, 300, 400]
    g = curves[(curves["n_estimators"].isin(checkpoints)) &
               (curves["start_weights"] == chosen["start_weights"])]
    pv = g.pivot(index="learning_rate", columns="n_estimators", values="val_avg_precision")
    fig, ax = plt.subplots(figsize=(8.6, 4))
    ax.imshow(pv.values, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(len(pv.columns)), pv.columns)
    ax.set_yticks(range(len(pv.index)), [f"{v:g}" for v in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color=SURFACE if v > np.nanpercentile(pv.values, 70) else INK)
    ax.set_xlabel("Boosting rounds (n_estimators)")
    ax.set_ylabel("Learning rate")
    ax.set_title(f"Validation average precision across both settings "
                 f"({chosen['start_weights']} start)", fontsize=12)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "ada_fig_tuning_grid.png")


def fig_confusion(yte, pred, title, fname):
    labels = ["No crash", "Crash"]
    cm = confusion_matrix(yte, pred, labels=labels)
    acc = accuracy_score(yte, pred)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(cm, cmap=SEQ)
    ax.set_xticks([0, 1], [f"Predicted\n{l}" for l in labels])
    ax.set_yticks([0, 1], [f"Actual\n{l}" for l in labels])
    tot = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}\n({v/tot:.1%})", ha="center", va="center",
                    fontsize=13, color=SURFACE if v > cm.max() * 0.5 else INK)
    ax.set_title(f"{title}\naccuracy {acc:.1%}", fontsize=12.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, fname)
    return cm


def fig_vs_tree(res):
    r = res[~res["model"].isin(["Majority-class baseline", "AdaBoost at the default cut-off"])]
    metrics_ = ["accuracy", "precision_crash", "recall_crash", "f1_crash"]
    nice = ["Accuracy", "Precision\n(Crash)", "Recall\n(Crash)", "F1\n(Crash)"]
    x = np.arange(len(metrics_))
    w = 0.8 / len(r)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for i, (_, row) in enumerate(r.iterrows()):
        ax.bar(x + (i - (len(r) - 1) / 2) * w, [row[m] * 100 for m in metrics_], w,
               color=SLOT[i], label=row["model"], edgecolor=SURFACE, linewidth=1.5)
    base = res[res["model"] == "Majority-class baseline"]["accuracy"].iloc[0] * 100
    ax.axhline(base, color=INK_SOFT, lw=1.2)
    ax.text(len(metrics_) - 0.5, base + 1.5, f"majority-class baseline accuracy {base:.1f}%",
            ha="right", fontsize=9.5, color=INK_SOFT)
    ax.set_xticks(x, nice)
    ax.set_ylabel("Percent, on the same 20,154 test complaints")
    ax.set_title("AdaBoost against the Part 3 decision tree")
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.set_ylim(0, 112)
    frame(ax)
    save(fig, "ada_fig_vs_tree.png")


def fig_alphas(model, Xcols):
    a = model.estimator_weights_[:len(model.estimators_)]
    e = model.estimator_errors_[:len(model.estimators_)]
    feats = [Xcols[s.tree_.feature[0]] if s.tree_.node_count > 1 else "(no split)"
             for s in model.estimators_]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    a1.bar(np.arange(1, len(a) + 1), a, color=SLOT[0], width=1.0)
    a1.set_xlabel("Boosting round")
    a1.set_ylabel("Stump weight alpha")
    a1.set_title("Each stump's say in the final vote", fontsize=11.5)
    frame(a1)
    vote = pd.Series(a, index=feats).groupby(level=0).sum().sort_values().tail(12)
    a2.barh([v.replace("_", " ") for v in vote.index], vote.values / a.sum() * 100,
            color=SLOT[3], height=0.68)
    a2.set_xlabel("Share of the total vote (%)")
    a2.set_title("Which features the stumps split on", fontsize=11.5)
    frame(a2, axis="x"); a2.grid(axis="x"); a2.grid(axis="y", visible=False)
    save(fig, "ada_fig_stump_weights.png")
    return pd.DataFrame({"round": np.arange(1, len(a) + 1), "split_feature": feats,
                         "threshold": [s.tree_.threshold[0] for s in model.estimators_],
                         "weighted_error": e, "alpha": a})


# ----- concept diagrams -------------------------------------------------

def concept_bagging_vs_boosting():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.9))
    for ax in (a1, a2):
        ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(ax, x, y, w, h, t, fc, tc=INK, fs=9.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=GRID, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(ax, x1, y1, x2, y2, c=INK_SOFT):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5))

    # Bagging: parallel
    box(a1, 3.5, 8.4, 3, 1.1, "Training data", "#f5f7fa")
    for i, x in enumerate([0.4, 3.6, 6.8]):
        arrow(a1, 5, 8.35, x + 1.4, 7.05)
        box(a1, x, 5.9, 2.8, 1.1, f"random sample {i+1}", "#eef5fd", fs=9)
        arrow(a1, x + 1.4, 5.85, x + 1.4, 4.75)
        box(a1, x, 3.6, 2.8, 1.1, f"tree {i+1}\n(full depth)", "#eef5fd", fs=9)
        arrow(a1, x + 1.4, 3.55, 5, 2.35)
    box(a1, 2.5, 1.2, 5, 1.1, "equal vote", "#eaf7f1", SLOT[2], 10.5)
    a1.text(5, 0.35, "trees built independently, at the same time", ha="center",
            fontsize=9.5, color=INK_SOFT, style="italic")
    a1.set_title("Bagging (e.g. random forest)", fontsize=12)

    # Boosting: sequential
    xs = [0.2, 3.5, 6.8]
    for i, x in enumerate(xs):
        box(a2, x, 6.6, 3.0, 1.3, f"round {i+1}: stump\nfit to weighted data", "#eef5fd", fs=9)
        a2.text(x + 1.5, 8.35, ["equal weights", "misses up-weighted", "misses up-weighted"][i],
                ha="center", fontsize=8.8, color=SIGNAL if i else INK_SOFT)
        if i < 2:
            arrow(a2, x + 3.02, 7.25, xs[i + 1] - 0.02, 7.25, SIGNAL)
        arrow(a2, x + 1.5, 6.55, 5, 2.35)
        a2.text(x + 1.5 + (0.25 if i == 0 else 0), 4.6, f"α{'₁₂₃'[i]}",
                fontsize=12, color=SLOT[3], ha="center")
    box(a2, 2.5, 1.2, 5, 1.1, "weighted vote: sign(Σ αₘ hₘ(x))", "#fdecef", SIGNAL, 10.5)
    a2.text(5, 0.35, "each stump built after, and because of, the previous one",
            ha="center", fontsize=9.5, color=INK_SOFT, style="italic")
    a2.set_title("Boosting (AdaBoost)", fontsize=12)
    save(fig, "ada_concept_bagging_vs_boosting.png")


def concept_reweighting():
    """Three real rounds of AdaBoost (SAMME, two classes) on toy data,
    computed by hand so the reweighting is visible."""
    rng = np.random.default_rng(8)
    n = 60
    X = rng.uniform(0, 10, (n, 2))
    y = np.where((X[:, 0] > 4) & (X[:, 1] > 3.5), 1, -1)
    flip = rng.random(n) < 0.08
    y[flip] *= -1
    w = np.full(n, 1 / n)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
    F = np.zeros(n)
    for r in range(3):
        ax = axes[r]
        st = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        h = st.predict(X)
        miss = h != y
        err = w[miss].sum() / w.sum()
        alpha = np.log((1 - err) / err)
        f, t = st.tree_.feature[0], st.tree_.threshold[0]
        size = 30 + 2600 * w
        for cls, col, mk in ((-1, SLOT[0], "o"), (1, SIGNAL, "^")):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=size[m], color=col, marker=mk, alpha=0.75,
                       edgecolor=SURFACE, linewidth=0.8)
        ax.scatter(X[miss, 0], X[miss, 1], s=size[miss] + 60, facecolor="none",
                   edgecolor=INK, linewidth=1.3)
        if f == 0:
            ax.axvline(t, color=INK, lw=2)
        else:
            ax.axhline(t, color=INK, lw=2)
        ax.set_title(f"Round {r+1}: error {err:.2f}, α = {alpha:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        F += alpha * h
        w = w * np.exp(alpha * miss)
        w = w / w.sum()

    ax = axes[3]
    pred = np.sign(F)
    for cls, col, mk in ((-1, SLOT[0], "o"), (1, SIGNAL, "^")):
        m = pred == cls
        ax.scatter(X[m, 0], X[m, 1], s=40, color=col, marker=mk, alpha=0.85,
                   edgecolor=SURFACE, linewidth=0.8)
    ax.set_title(f"Weighted vote of 3 stumps: {np.mean(pred == y):.0%} correct", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.suptitle("Marker size = row weight. Circled rows were misclassified and grow "
                 "before the next round.", fontsize=11, color=INK_SOFT, y=1.02)
    save(fig, "ada_concept_reweighting.png")


def concept_alpha():
    e = np.linspace(0.01, 0.99, 300)
    a = np.log((1 - e) / e)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(e, a, color=SLOT[3], lw=2.4)
    ax.axhline(0, color=INK_SOFT, lw=1)
    ax.axvline(0.5, color=GRID, lw=1.2)
    ax.text(0.06, 3.2, "accurate stump\n→ large say", fontsize=10, color=INK)
    ax.text(0.53, 0.35, "coin-flip stump (error 0.5)\n→ no say at all", fontsize=10, color=INK)
    ax.text(0.7, -3.0, "worse than chance\n→ vote is reversed", fontsize=10, color=INK)
    ax.set_xlabel("Weighted error of the stump in that round")
    ax.set_ylabel("α = learning rate × ln((1 − error) / error)")
    ax.set_title("How a stump's error sets its weight in the vote")
    frame(ax, axis="both")
    save(fig, "ada_concept_alpha.png")


# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("ADABOOST  --  predicting a crash from structured attributes")
    print("=" * 72)

    before, feat = build_features()
    Xtr, Xte, ytr, yte, Xfit, Xval, yfit, yval = split(feat)
    curves, chosen, thr_curve = tune(Xfit, yfit, Xval, yval)

    print("\n[4] FINAL MODEL: refit on the whole training set, scored once on test")
    lr, n, thr = chosen["learning_rate"], chosen["n_estimators"], chosen["threshold"]
    w = compute_sample_weight("balanced", ytr) if chosen["start_weights"] == "balanced" else None
    ada = stumps(n, lr).fit(Xtr, ytr, sample_weight=w)
    s_te = crash_score(ada, Xte)
    p_ada = np.where(s_te > thr, "Crash", "No crash")
    p_ada_default = np.where(s_te > 0, "Crash", "No crash")
    print(f"    test average precision (threshold-free): "
          f"{average_precision_score((yte == 'Crash').astype(int), s_te):.4f}")

    # The Part 3 tree that scored best, refitted with identical settings.
    tree = DecisionTreeClassifier(criterion="gini", max_depth=3, class_weight="balanced",
                                  random_state=RANDOM_STATE).fit(Xtr, ytr)
    p_tree = tree.predict(Xte)
    stump = DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                                   random_state=RANDOM_STATE).fit(Xtr, ytr)
    p_stump = stump.predict(Xte)
    p_maj = np.full(len(yte), ytr.value_counts().idxmax())

    res = pd.DataFrame([metrics(f"AdaBoost ({n} stumps, lr {lr:g})", yte, p_ada),
                        metrics("Part 3 decision tree (gini, depth 3)", yte, p_tree),
                        metrics("AdaBoost at the default cut-off", yte, p_ada_default),
                        metrics("A single stump", yte, p_stump),
                        metrics("Majority-class baseline", yte, p_maj)]).round(4)
    print(res.to_string(index=False))
    pub = pd.read_csv(CLEAN / "dt_results.csv").set_index("model").loc["Tree 1: gini, depth 3"]
    got = res.set_index("model").loc["Part 3 decision tree (gini, depth 3)"]
    assert abs(got["f1_crash"] - pub["f1_crash"]) < 1e-3 and \
        abs(got["accuracy"] - pub["accuracy"]) < 1e-3, "Part 3 tree does not reproduce"
    print("    Part 3 tree reproduces its published result exactly")
    res.to_csv(CLEAN / "ada_results.csv", index=False)

    d_f1 = res.iloc[0]["f1_crash"] - res.iloc[1]["f1_crash"]
    d_acc = res.iloc[0]["accuracy"] - res.iloc[1]["accuracy"]
    print(f"    AdaBoost minus tree: F1 {d_f1:+.4f}, accuracy {d_acc:+.4f}")

    print("\n[5] FIGURES")
    concept_bagging_vs_boosting()
    concept_reweighting()
    concept_alpha()

    table_png(before.drop(columns=["model_year", "n_recalls"]),
              f"BEFORE encoding: {len(before):,} labelled complaints, 'segment' and "
              "'component' are still text", "ada_sample_before.png", width=14)
    show = ["vehicle_age", "days_to_report", "overall_rating", "rollover_prob", "has_fcw",
            "seg_Electric", "seg_SUV", "seg_Sedan", "cmp_AIR BAGS", "cmp_ENGINE",
            "cmp_SERVICE BRAKES", "cmp_OTHER", "label"]
    table_png(feat[show], f"AFTER one-hot encoding: every category is its own 0/1 column, "
              f"{feat.shape[1]-1} numeric features (a subset shown)",
              "ada_sample_encoded.png")
    table_png(Xtr.assign(label=ytr)[show],
              f"TRAINING set: {len(Xtr):,} rows ({1-TEST_SIZE:.0%}); "
              f"{len(Xval):,} of them held out again for validation",
              "ada_sample_train.png")
    table_png(Xte.assign(label=yte)[show],
              f"TESTING set: {len(Xte):,} rows ({TEST_SIZE:.0%}), the same rows the "
              "Part 3 tree was tested on", "ada_sample_test.png")

    fig_curves(curves, chosen)
    fig_threshold(thr_curve, chosen)
    fig_grid(curves, chosen)
    fig_confusion(yte, p_ada, f"AdaBoost, {n} stumps, learning rate {lr:g}",
                  "ada_fig_confusion.png")
    fig_vs_tree(res)
    st = fig_alphas(ada, list(Xtr.columns))
    st.round(5).to_csv(CLEAN / "ada_stumps.csv", index=False)
    print("\n    first five stumps:")
    print(st.head(5).round(4).to_string(index=False))
    print("done.")
