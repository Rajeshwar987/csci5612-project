"""
decision_tree_analysis.py  --  CSCI 5612 Project, Module 3 Part 3
Supervised classification with Decision Trees.

Task: the same label as the Naive Bayes tab - did this complaint involve a
crash - but predicted from the structured attributes of the vehicle and the
report rather than from the narrative text. Keeping the target identical makes
the two methods directly comparable.

Outputs:
    data/clean/dt_labeled_sample.csv   the labelled feature matrix
    data/clean/dt_train_sample.csv     a sample of the training set
    data/clean/dt_test_sample.csv      a sample of the testing set
    data/clean/dt_results.csv          metrics for all three trees
    data/clean/dt_information_gain.csv the worked split example
    figures/dt_*.png                   concepts, three trees, matrices
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 5612
TEST_SIZE = 0.25

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


def table_png(df, title, fname, nrows=6, width=14, fs=7.4):
    sub = df.head(nrows).copy()
    for c in sub.columns:
        sub[c] = sub[c].astype(str).str.slice(0, 26)
    fig, ax = plt.subplots(figsize=(width, 0.42 * nrows + 1.4))
    ax.axis("off")
    t = ax.table(cellText=sub.values,
                 colLabels=[c.replace("_", " ")[:18] for c in sub.columns],
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.45)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#f5f7fa")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK_SOFT)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=14)
    save(fig, fname)


# ======================================================================
# IMPURITY MATHS  (used for the worked example)
# ======================================================================

def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return float(1 - (p ** 2).sum())


def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_gain(y, mask, fn):
    n = len(y)
    left, right = y[mask], y[~mask]
    weighted = len(left) / n * fn(left) + len(right) / n * fn(right)
    return fn(y) - weighted, fn(y), weighted, len(left), len(right)

def build_features():
    print("\n[1] LABELLED DATA")
    c = pd.read_csv(CLEAN / "complaints_clean.csv.gz", low_memory=False)
    my = pd.read_csv(CLEAN / "vehicle_model_year.csv")

    # Join the vehicle-level safety attributes onto each complaint.
    cols = ["make", "model", "model_year", "OverallRating", "RolloverPossibility",
            "has_fcw", "has_ldw", "n_recalls"]
    c = c.merge(my[cols], on=["make", "model", "model_year"], how="inner")
    print(f"    complaints joined to rated vehicles: {len(c):,}")

    c["label"] = np.where(c["crash"], "Crash", "No crash")

    # Keep the most common components as named categories; the long tail goes
    # to 'Other' so the one-hot encoding does not explode.
    top = c["component_primary"].value_counts().head(12).index
    c["component"] = np.where(c["component_primary"].isin(top),
                              c["component_primary"], "OTHER")

    feat = pd.DataFrame({
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

    feat = pd.concat([feat,
                      pd.get_dummies(c["segment"], prefix="seg").astype(int),
                      pd.get_dummies(c["component"], prefix="cmp").astype(int)],
                     axis=1)
    feat["label"] = c["label"].values
    feat = feat.dropna().reset_index(drop=True)

    print(f"    feature matrix: {feat.shape[0]:,} rows x {feat.shape[1]-1} features")
    print(f"    class balance: {feat['label'].value_counts(normalize=True).round(4).to_dict()}")
    feat.head(500).to_csv(CLEAN / "dt_labeled_sample.csv", index=False)
    return feat, c

def split(feat):
    print("\n[2] TRAIN / TEST SPLIT")
    X = feat.drop(columns=["label"])
    y = feat["label"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    shared = set(Xtr.index) & set(Xte.index)
    print(f"    train: {len(Xtr):,}   test: {len(Xte):,}")
    print(f"    rows appearing in both sets: {len(shared)}")
    assert not shared, "train and test are not disjoint"
    print(f"    train balance: {ytr.value_counts(normalize=True).round(4).to_dict()}")
    print(f"    test  balance: {yte.value_counts(normalize=True).round(4).to_dict()}")

    Xtr.assign(label=ytr).head(400).to_csv(CLEAN / "dt_train_sample.csv", index=False)
    Xte.assign(label=yte).head(400).to_csv(CLEAN / "dt_test_sample.csv", index=False)
    return Xtr, Xte, ytr, yte



def worked_example(Xtr, ytr):
    print("\n[3] WORKED SPLIT EXAMPLE (real training data)")
    y = (ytr == "Crash").astype(int).values

    candidates = [
        ("summary_words <= 40", Xtr["summary_words"].values <= 40),
        ("vehicle_age <= 2", Xtr["vehicle_age"].values <= 2),
        ("has_fcw == 1", Xtr["has_fcw"].values == 1),
        ("rollover_prob <= 0.15", Xtr["rollover_prob"].values <= 0.15),
    ]
    rows = []
    for name, mask in candidates:
        g_gain, g_parent, g_child, nl, nr = info_gain(y, mask, gini)
        e_gain, e_parent, e_child, _, _ = info_gain(y, mask, entropy)
        rows.append({"split": name, "n_left": nl, "n_right": nr,
                     "gini_parent": round(g_parent, 5),
                     "gini_weighted_children": round(g_child, 5),
                     "gini_gain": round(g_gain, 6),
                     "entropy_parent": round(e_parent, 5),
                     "entropy_weighted_children": round(e_child, 5),
                     "information_gain": round(e_gain, 6)})
        print(f"    {name:<24} gini gain={g_gain:.6f}   info gain={e_gain:.6f}")

    tab = pd.DataFrame(rows).sort_values("information_gain", ascending=False)
    tab.to_csv(CLEAN / "dt_information_gain.csv", index=False)
    best = tab.iloc[0]
    print(f"    -> the tree would choose: {best['split']}")
    return tab, best



def fit_trees(Xtr, ytr, Xte, yte):
    print("\n[4] THREE TREES")
    specs = [
        ("Tree 1: gini, depth 3",
         DecisionTreeClassifier(criterion="gini", max_depth=3,
                                class_weight="balanced", random_state=RANDOM_STATE)),
        ("Tree 2: entropy, depth 5",
         DecisionTreeClassifier(criterion="entropy", max_depth=5,
                                class_weight="balanced", random_state=RANDOM_STATE)),
        ("Tree 3: gini, depth 10, min_leaf 50",
         DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=50,
                                class_weight="balanced", random_state=RANDOM_STATE)),
    ]
    out, rows = [], []
    for name, clf in specs:
        clf.fit(Xtr, ytr)
        p = clf.predict(Xte)
        acc = accuracy_score(yte, p)
        rows.append({"model": name, "accuracy": acc,
                     "precision_crash": precision_score(yte, p, pos_label="Crash", zero_division=0),
                     "recall_crash": recall_score(yte, p, pos_label="Crash", zero_division=0),
                     "f1_crash": f1_score(yte, p, pos_label="Crash", zero_division=0),
                     "leaves": clf.get_n_leaves(), "depth": clf.get_depth()})
        out.append((name, clf, p))
        print(f"    {name:<36} acc={acc:.4f}  F1(Crash)={rows[-1]['f1_crash']:.4f}  "
              f"leaves={clf.get_n_leaves()}")

    maj = pd.Series(ytr).value_counts().idxmax()
    pmaj = np.full(len(yte), maj)
    rows.append({"model": "Majority-class baseline",
                 "accuracy": accuracy_score(yte, pmaj),
                 "precision_crash": 0.0, "recall_crash": 0.0, "f1_crash": 0.0,
                 "leaves": 1, "depth": 0})

    res = pd.DataFrame(rows).round(4)
    print("\n" + res.to_string(index=False))
    print("\n  best tree, full report:")
    best_i = int(np.argmax([r["f1_crash"] for r in rows[:3]]))
    print(classification_report(yte, out[best_i][2], zero_division=0))
    return out, res, best_i


def depth_curve(Xtr, ytr, Xte, yte):
    print("\n[5] DEPTH SWEEP (overfitting check)")
    rows = []
    for d in range(1, 21):
        clf = DecisionTreeClassifier(criterion="gini", max_depth=d,
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE).fit(Xtr, ytr)
        rows.append({"depth": d,
                     "train_f1": f1_score(ytr, clf.predict(Xtr), pos_label="Crash", zero_division=0),
                     "test_f1": f1_score(yte, clf.predict(Xte), pos_label="Crash", zero_division=0)})
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    return df



def fig_tree(clf, Xtr, name, fname, fs=9):
    fig, ax = plt.subplots(figsize=(17, 8.5))
    plot_tree(clf, feature_names=[c.replace("_", " ") for c in Xtr.columns],
              class_names=clf.classes_, filled=True, rounded=True,
              impurity=True, proportion=True, fontsize=fs, ax=ax,
              max_depth=3)
    ax.set_title(f"{name}  (showing the top 3 levels; "
                 f"full tree has {clf.get_n_leaves()} leaves)",
                 fontsize=13, fontweight="bold", color=INK)
    save(fig, fname)


def fig_confusion(yte, pred, title, fname, acc):
    labels = ["No crash", "Crash"]
    cm = confusion_matrix(yte, pred, labels=labels)
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


def fig_importance(clf, Xtr, n=12):
    imp = pd.Series(clf.feature_importances_, index=Xtr.columns)
    imp = imp.sort_values().tail(n)
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.barh([i.replace("_", " ") for i in imp.index], imp.values, color=SLOT[0], height=0.7)
    ax.set_title("Which attributes the tree actually used")
    ax.set_xlabel("Feature importance (share of impurity reduction)")
    frame(ax, axis="x"); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    save(fig, "dt_fig_importance.png")


def fig_depth(df):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(df["depth"], df["train_f1"] * 100, color=SLOT[0], lw=2, marker="o",
            ms=4.5, mfc=SLOT[0], mec=SURFACE, mew=1.3, label="Training set")
    ax.plot(df["depth"], df["test_f1"] * 100, color=SIGNAL, lw=2, marker="s",
            ms=4.5, mfc=SIGNAL, mec=SURFACE, mew=1.3, label="Testing set")
    best = df.loc[df["test_f1"].idxmax()]
    ax.axvline(best["depth"], color=INK_SOFT, lw=1)
    ax.annotate(f"best test F1 at depth {int(best['depth'])}",
                xy=(best["depth"], best["test_f1"] * 100),
                xytext=(8, 12), textcoords="offset points", fontsize=10, color=INK)
    ax.set_title("Why an unlimited tree is not the best tree")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("F1 on the Crash class (%)")
    ax.set_xticks(range(0, 21, 2))
    ax.legend(loc="upper left")
    frame(ax)
    save(fig, "dt_fig_depth_curve.png")


def fig_impurity_curves():
    p = np.linspace(0.001, 0.999, 400)
    g = 1 - (p ** 2 + (1 - p) ** 2)
    e = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(p, e, color=SLOT[0], lw=2.2, label="Entropy (max 1.0)")
    ax.plot(p, g, color=SLOT[1], lw=2.2, label="Gini impurity (max 0.5)")
    ax.axvline(0.5, color=GRID, lw=1.2)
    ax.text(0.5, 1.03, "50/50 split:\nmaximum impurity", ha="center",
            fontsize=9.5, color=INK_SOFT)
    ax.text(0.02, 0.06, "all one class:\nzero impurity", fontsize=9.5, color=INK_SOFT)
    ax.set_title("Two ways of measuring how mixed a group is")
    ax.set_xlabel("Proportion of the group belonging to one class")
    ax.set_ylabel("Impurity")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right")
    frame(ax)
    save(fig, "dt_concept_impurity.png")


def fig_worked_example(tab, best):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    t = tab.sort_values("information_gain")
    a1.barh(t["split"], t["information_gain"], color=SLOT[0], height=0.6,
            label="Information gain (entropy)")
    a1.barh(t["split"], t["gini_gain"], color=SLOT[1], height=0.3,
            label="Gini gain")
    a1.set_title("Four candidate splits, scored on the real training data", fontsize=11.5)
    a1.set_xlabel("Gain (higher is a better split)")
    a1.legend(loc="lower right")
    frame(a1, axis="x"); a1.grid(axis="x"); a1.grid(axis="y", visible=False)

    a2.axis("off")
    a2.set_xlim(0, 10); a2.set_ylim(0, 10)
    a2.text(5, 9.4, f"Worked example: {best['split']}", ha="center",
            fontsize=12, fontweight="bold", color=INK)
    lines = [
        ("Parent group", f"entropy = {best['entropy_parent']:.5f}"),
        ("", ""),
        (f"Left branch  (n = {int(best['n_left']):,})", "matches the rule"),
        (f"Right branch (n = {int(best['n_right']):,})", "does not match"),
        ("", ""),
        ("Weighted children", f"entropy = {best['entropy_weighted_children']:.5f}"),
        ("", ""),
        ("Information gain", f"{best['entropy_parent']:.5f} - "
                             f"{best['entropy_weighted_children']:.5f}"),
        ("", f"= {best['information_gain']:.6f}"),
        ("", ""),
        ("Same split by Gini", f"{best['gini_parent']:.5f} - "
                               f"{best['gini_weighted_children']:.5f} = "
                               f"{best['gini_gain']:.6f}"),
    ]
    y = 8.4
    for left, right in lines:
        if left:
            a2.text(0.3, y, left, fontsize=10, color=INK, fontweight="bold")
        if right:
            a2.text(9.7, y, right, fontsize=10, color=INK_SOFT, ha="right")
        y -= 0.72
    a2.text(5, 0.3, "The tree tries every rule like this and keeps the one with the\n"
                    "largest gain, then repeats inside each branch.",
            ha="center", fontsize=9.5, color=INK_SOFT, style="italic")
    save(fig, "dt_concept_worked_example.png")


def concept_how_dt_works():
    rng = np.random.default_rng(4)
    n = 170
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    cls = ((x > 5.5) & (y > 4)).astype(int)
    flip = rng.random(n) < 0.08
    cls[flip] = 1 - cls[flip]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.9))

    for c, col, mk, lab in ((0, SLOT[0], "o", "No crash"), (1, SIGNAL, "^", "Crash")):
        m = cls == c
        a1.scatter(x[m], y[m], s=40, color=col, marker=mk, alpha=0.8,
                   edgecolor=SURFACE, linewidth=0.8, label=lab)
    a1.axvline(5.5, color=INK, lw=2)
    a1.plot([5.5, 10], [4, 4], color=INK, lw=2)
    a1.text(2.6, 9.2, "split 1", fontsize=10, color=INK)
    a1.text(7.6, 4.3, "split 2", fontsize=10, color=INK)
    a1.set_title("A tree carves the data into rectangles", fontsize=11.5)
    a1.set_xlabel("Feature A"); a1.set_ylabel("Feature B")
    a1.legend(loc="upper left")
    frame(a1, axis="both")

    a2.axis("off")
    a2.set_xlim(0, 10); a2.set_ylim(0, 10)

    def node(cx, cy, w, h, txt, fc, tc=INK):
        a2.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                                   edgecolor=GRID, linewidth=1.2))
        a2.text(cx, cy, txt, ha="center", va="center", fontsize=9.5, color=tc)

    node(5, 8.6, 3.6, 1.3, "Feature A > 5.5 ?", "#eef5fd")
    node(2.2, 5.6, 3.0, 1.3, "No crash", "#eaf1fb", SLOT[0])
    node(7.4, 5.6, 3.4, 1.3, "Feature B > 4 ?", "#eef5fd")
    node(5.6, 2.4, 2.8, 1.3, "No crash", "#eaf1fb", SLOT[0])
    node(9.0, 2.4, 2.4, 1.3, "Crash", "#fdecef", SIGNAL)
    for (x1, y1), (x2, y2) in (((5, 7.95), (2.2, 6.25)), ((5, 7.95), (7.4, 6.25)),
                               ((7.4, 4.95), (5.6, 3.05)), ((7.4, 4.95), (9.0, 3.05))):
        a2.plot([x1, x2], [y1, y2], color=INK_SOFT, lw=1.4)
    a2.text(3.2, 7.3, "no", fontsize=9, color=INK_SOFT)
    a2.text(6.6, 7.3, "yes", fontsize=9, color=INK_SOFT)
    a2.set_title("The same thing drawn as a tree", fontsize=11.5)
    save(fig, "dt_concept_how_it_works.png")


# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("DECISION TREES  --  predicting a crash from structured attributes")
    print("=" * 72)

    feat, raw = build_features()
    Xtr, Xte, ytr, yte = split(feat)
    tab, best = worked_example(Xtr, ytr)
    trees, res, best_i = fit_trees(Xtr, ytr, Xte, yte)
    dcurve = depth_curve(Xtr, ytr, Xte, yte)

    res.to_csv(CLEAN / "dt_results.csv", index=False)
    dcurve.round(5).to_csv(CLEAN / "dt_depth_sweep.csv", index=False)

    print("\n[6] FIGURES")
    concept_how_dt_works()
    fig_impurity_curves()
    fig_worked_example(tab, best)

    table_png(feat, f"Labelled feature matrix: {len(feat):,} rows x "
                    f"{feat.shape[1]-1} numeric features, label in the last column",
              "dt_sample_labeled.png", width=15, fs=6.6)
    table_png(Xtr.assign(label=ytr),
              f"TRAINING set: {len(Xtr):,} rows ({1-TEST_SIZE:.0%})",
              "dt_sample_train.png", width=15, fs=6.6)
    table_png(Xte.assign(label=yte),
              f"TESTING set: {len(Xte):,} rows ({TEST_SIZE:.0%}), never seen in training",
              "dt_sample_test.png", width=15, fs=6.6)

    for i, (name, clf, pred) in enumerate(trees, 1):
        fig_tree(clf, Xtr, name, f"dt_fig_tree{i}.png")

    bname, bclf, bpred = trees[best_i]
    fig_confusion(yte, bpred, bname, "dt_fig_confusion.png", accuracy_score(yte, bpred))
    fig_importance(bclf, Xtr)
    fig_depth(dcurve)

    print(f"\n  best tree by F1: {bname}")
    print("done.")