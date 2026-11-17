"""
regression_analysis.py  --  CSCI 5612 Project, Module 4 Part 4
Linear and logistic regression.

Two separate questions, one per model:

  LINEAR   (numeric target)  For each model-year with at least 20 complaints,
           what share of its complaints involve a crash? Predicted from the
           vehicle's crash-test ratings, equipment and ownership record.

  LOGISTIC (two-class label) Did this complaint involve a crash? Predicted
           from the narrative text, on exactly the same complaints, the same
           word-count matrix and the same train/test split as the Naive Bayes
           tab, so the two classifiers can be compared directly.

Settings are chosen with cross-validation inside the training set only. The
test set is used once, at the end, to score the fixed models.

Outputs:
    data/clean/reg_linear_*.csv       data, split, CV grid, coefficients, metrics
    data/clean/reg_logistic_*.csv     split samples, CV grid, coefficients, metrics
    figures/reg_*.png                 concepts, data images, results
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 5612
TEST_SIZE = 0.25
MIN_COMPLAINTS = 20       # below this a crash *rate* is mostly noise

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


def _fit(s, max_pts, fs):
    """Cut a string at a word boundary so it renders within max_pts."""
    if _text_pts(s, fs) <= max_pts:
        return s
    words, out = s.split(), ""
    for w in words:
        cand = (out + " " + w).strip()
        if _text_pts(cand + "…", fs) > max_pts:
            break
        out = cand
    return out.rstrip(",.;:") + "…"


def table_png(df, title, fname, nrows=6, width=None, fs=8.4, text_pts=430):
    """Small image of the first rows of a table. Every column is exactly as
    wide as its widest rendered entry, long text is left-aligned and cut at a
    word boundary, and the image is sized to the table itself, so there is no
    empty space inside cells or around the table."""
    sub = df.head(nrows).copy()
    for c in sub.columns:
        sub[c] = sub[c].map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
        sub[c] = sub[c].map(lambda s: " ".join(s.split()))
    labels = [c.replace("_", " ") for c in sub.columns]
    text_cols = {j for j, c in enumerate(sub.columns)
                 if max(_text_pts(v, fs) for v in sub[c]) > text_pts}
    for j in text_cols:
        c = sub.columns[j]
        sub[c] = sub[c].map(lambda s: _fit(s, text_pts, fs))
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
            cell.PAD = (pad / 2) / col_pts[j]
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


def clf_metrics(name, y, p):
    return {"model": name,
            "accuracy": accuracy_score(y, p),
            "precision_crash": precision_score(y, p, pos_label="Crash", zero_division=0),
            "recall_crash": recall_score(y, p, pos_label="Crash", zero_division=0),
            "f1_crash": f1_score(y, p, pos_label="Crash", zero_division=0)}


# ======================================================================
# PART A  --  LINEAR REGRESSION
# ======================================================================

LAB = ["OverallRating", "OverallFrontCrashRating", "OverallSideCrashRating",
       "RolloverRating", "rollover_prob_pct"]
EQUIP = ["has_fcw", "has_ldw"]
RECORD = ["mean_vehicle_age", "median_days_to_report", "log_complaints", "log_recalls"]

NICE = {
    "OverallRating": "Overall stars", "OverallFrontCrashRating": "Front-crash stars",
    "OverallSideCrashRating": "Side-crash stars", "RolloverRating": "Rollover stars",
    "rollover_prob_pct": "Rollover probability (pct points)",
    "has_fcw": "Collision warning standard", "has_ldw": "Lane warning standard",
    "mean_vehicle_age": "Mean vehicle age at incident (yrs)",
    "median_days_to_report": "Median days to report",
    "log_complaints": "log(complaint count)", "log_recalls": "log(recall count)",
}


def linear_data():
    print("\n[A1] LINEAR REGRESSION DATA")
    my = pd.read_csv(CLEAN / "vehicle_model_year.csv")
    n0 = len(my)
    my = my[my["n_complaints"] >= MIN_COMPLAINTS].reset_index(drop=True)
    print(f"    model-years with >= {MIN_COMPLAINTS} complaints: {n0} -> {len(my)}")

    d = pd.DataFrame({
        "make": my["make"], "model": my["model"], "model_year": my["model_year"],
        "segment": my["segment"],
        "OverallRating": my["OverallRating"],
        "OverallFrontCrashRating": my["OverallFrontCrashRating"],
        "OverallSideCrashRating": my["OverallSideCrashRating"],
        "RolloverRating": my["RolloverRating"],
        "rollover_prob_pct": (my["RolloverPossibility"] * 100).round(2),
        "has_fcw": my["has_fcw"], "has_ldw": my["has_ldw"],
        "mean_vehicle_age": my["mean_vehicle_age"],
        "median_days_to_report": my["median_days_to_report"],
        "log_complaints": np.log1p(my["n_complaints"]).round(4),
        "log_recalls": np.log1p(my["n_recalls"]).round(4),
    })
    # Body segment is categorical, so it is one-hot encoded; SUV is the
    # reference level (dropped) so each segment coefficient reads as
    # "relative to an SUV".
    seg = pd.get_dummies(d["segment"], prefix="seg").astype(int).drop(columns="seg_SUV")
    d = pd.concat([d, seg], axis=1)
    d["crash_rate_pct"] = (my["crash_rate"] * 100).round(3)
    SEG = list(seg.columns)
    print(f"    target crash_rate_pct: mean {d['crash_rate_pct'].mean():.2f}, "
          f"sd {d['crash_rate_pct'].std():.2f}")
    d.to_csv(CLEAN / "reg_linear_data.csv", index=False)
    return d, SEG


def linear_split(d):
    print("\n[A2] TRAIN / TEST SPLIT")
    tr, te = train_test_split(d, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    keys = ["make", "model", "model_year"]
    shared = pd.merge(tr[keys], te[keys], on=keys)
    print(f"    train {len(tr)}   test {len(te)}   vehicles in both: {len(shared)}")
    assert shared.empty, "train and test overlap"
    tr.to_csv(CLEAN / "reg_linear_train.csv", index=False)
    te.to_csv(CLEAN / "reg_linear_test.csv", index=False)
    return tr, te


def linear_cv(tr, SEG):
    """Two settings chosen by 5-fold cross-validation on the TRAINING set:
    (1) which group of features the model may use, (2) the ridge penalty
    alpha, where alpha = 0 is ordinary least squares (plain LinearRegression)."""
    print("\n[A3] CROSS-VALIDATION ON THE TRAINING SET")
    sets = {
        "Crash-test ratings only": LAB,
        "Ratings + equipment + segment": LAB + EQUIP + SEG,
        "Ownership record only": RECORD,
        "Everything": LAB + EQUIP + SEG + RECORD,
    }
    alphas = [0, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    y = tr["crash_rate_pct"].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for sname, cols in sets.items():
        X = tr[cols].values
        for a in alphas:
            model = (make_pipeline(StandardScaler(), LinearRegression()) if a == 0
                     else make_pipeline(StandardScaler(), Ridge(alpha=a)))
            p = cross_val_predict(model, X, y, cv=kf)
            rows.append({"feature_set": sname, "alpha": a,
                         "cv_rmse": float(np.sqrt(mean_squared_error(y, p))),
                         "cv_r2": float(r2_score(y, p))})
    # Baseline: predict the training-fold mean.
    p0 = np.zeros_like(y, dtype=float)
    for a_idx, b_idx in kf.split(y):
        p0[b_idx] = y[a_idx].mean()
    base_rmse = float(np.sqrt(mean_squared_error(y, p0)))
    cv = pd.DataFrame(rows)
    print(cv.pivot(index="feature_set", columns="alpha", values="cv_rmse").round(3).to_string())
    print(f"    baseline (fold mean) CV RMSE: {base_rmse:.3f}")
    best = cv.loc[cv["cv_rmse"].idxmin()]
    print(f"    -> chosen: {best['feature_set']}, alpha={best['alpha']:g} "
          f"(CV RMSE {best['cv_rmse']:.3f}, CV R2 {best['cv_r2']:.3f})")
    cv.round(5).to_csv(CLEAN / "reg_linear_cv.csv", index=False)
    return cv, sets, best, base_rmse


def linear_final(tr, te, sets, best):
    print("\n[A4] FINAL LINEAR MODELS, SCORED ONCE ON THE TEST SET")
    cols = sets[best["feature_set"]]
    Xtr, Xte = tr[cols].values, te[cols].values
    ytr, yte = tr["crash_rate_pct"].values, te["crash_rate_pct"].values

    a = float(best["alpha"])
    chosen = (make_pipeline(StandardScaler(), LinearRegression()) if a == 0
              else make_pipeline(StandardScaler(), Ridge(alpha=a))).fit(Xtr, ytr)

    # Plain LinearRegression in original units on the same columns: this is
    # the model whose coefficients are read in plain words.
    ols = LinearRegression().fit(Xtr, ytr)

    rows = []
    preds = {}
    for name, m in (("Chosen model" + (f" (ridge, alpha={a:g})" if a else " (OLS)"), chosen),
                    ("LinearRegression (OLS), same features", ols)):
        p = m.predict(Xte)
        preds[name] = p
        rows.append({"model": name, "test_rmse": np.sqrt(mean_squared_error(yte, p)),
                     "test_mae": mean_absolute_error(yte, p), "test_r2": r2_score(yte, p)})
    p0 = np.full_like(yte, ytr.mean(), dtype=float)
    rows.append({"model": "Baseline: training mean", "test_rmse": np.sqrt(mean_squared_error(yte, p0)),
                 "test_mae": mean_absolute_error(yte, p0), "test_r2": r2_score(yte, p0)})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.round(4).to_csv(CLEAN / "reg_linear_results.csv", index=False)

    # Coefficients of the chosen model, converted back to original units
    # (coefficient / feature sd) so they read as "percentage points of crash
    # rate per one unit of the feature". The OLS coefficients are kept
    # alongside to show how collinear star ratings make them unstable.
    scaler, reg = chosen[0], chosen[-1]
    coef = pd.DataFrame({
        "feature": cols,
        "chosen_per_unit": reg.coef_ / scaler.scale_,
        "chosen_per_sd": reg.coef_,
        "ols_per_unit": ols.coef_,
    }).round(4)
    coef.to_csv(CLEAN / "reg_linear_coefficients.csv", index=False)
    print("\n    coefficients (pct points of crash rate):")
    print(coef.to_string(index=False))

    # Lab-only model scored on test too, for the topic question.
    lab = LinearRegression().fit(tr[LAB].values, ytr)
    pl = lab.predict(te[LAB].values)
    lab_row = {"test_rmse": np.sqrt(mean_squared_error(yte, pl)),
               "test_mae": mean_absolute_error(yte, pl), "test_r2": r2_score(yte, pl)}
    print(f"\n    ratings-only OLS on test: RMSE {lab_row['test_rmse']:.3f} "
          f"R2 {lab_row['test_r2']:.3f}")
    return chosen, ols, cols, preds, res, coef, lab_row, yte


# ----- linear figures ----------------------------------------------------

def fig_linear_cv(cv, base_rmse):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for i, (s, g) in enumerate(cv.groupby("feature_set", sort=False)):
        x = np.arange(len(g))
        ax.plot(x, g["cv_rmse"], color=SLOT[i], lw=2, marker="o", ms=5,
                mfc=SLOT[i], mec=SURFACE, mew=1.3, label=s)
    ax.axhline(base_rmse, color=INK_SOFT, lw=1.2)
    ax.text(len(g) - 1, base_rmse + 0.05, "baseline: always predict the average",
            ha="right", fontsize=9.5, color=INK_SOFT)
    ax.set_xticks(np.arange(len(g)), [f"{a:g}" for a in g["alpha"]])
    ax.set_xlabel("Ridge penalty alpha (0 = plain least squares)")
    ax.set_ylabel("Cross-validated RMSE (pct points)")
    ax.set_title("Choosing features and penalty on the training set only")
    ax.legend(loc="upper right", fontsize=9)
    frame(ax)
    save(fig, "reg_fig_linear_cv.png")


def fig_pred_vs_actual(yte, p, r2, rmse):
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lim = (0, max(yte.max(), p.max()) * 1.08)
    ax.plot(lim, lim, color=INK_SOFT, lw=1.2)
    ax.text(lim[1] * 0.97, lim[1] * 0.92, "perfect prediction", ha="right",
            fontsize=9.5, color=INK_SOFT, rotation=0)
    ax.scatter(yte, p, s=48, color=SLOT[0], alpha=0.8, edgecolor=SURFACE, linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual crash rate on the test vehicles (%)")
    ax.set_ylabel("Predicted crash rate (%)")
    ax.set_title(f"Predicted against actual, held-out test set\n"
                 f"R² = {r2:.3f}   RMSE = {rmse:.2f} pct points", fontsize=12)
    frame(ax, axis="both")
    save(fig, "reg_fig_pred_vs_actual.png")


def fig_coefficients(coef):
    # Per-SD coefficients put every bar on the same scale: the change in
    # predicted crash rate for a one-standard-deviation change in the feature.
    std_coef = pd.Series(coef["chosen_per_sd"].values,
                         index=[NICE.get(c, c.replace("seg_", "Segment: "))
                                for c in coef["feature"]]).sort_values()
    colors = [SIGNAL if v > 0 else SLOT[0] for v in std_coef.values]
    fig, ax = plt.subplots(figsize=(8.8, 0.36 * len(std_coef) + 1.6))
    ax.barh(std_coef.index, std_coef.values, color=colors, height=0.68)
    ax.axvline(0, color=INK_SOFT, lw=1)
    ax.set_title("What moves the predicted crash rate (chosen model)")
    ax.set_xlabel("Change in crash rate (pct points) for a one-SD increase in the feature")
    frame(ax, axis="x"); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    save(fig, "reg_fig_linear_coefficients.png")


# ======================================================================
# PART B  --  LOGISTIC REGRESSION (same data and split as Naive Bayes)
# ======================================================================

def logistic_data():
    print("\n[B1] LOGISTIC REGRESSION DATA (identical to the Naive Bayes tab)")
    c = pd.read_csv(CLEAN / "complaints_clean.csv.gz", low_memory=False)
    c["summary"] = c["summary"].fillna("").astype(str)
    c = c[c["summary"].str.split().str.len() >= 5]
    c["label"] = np.where(c["crash"], "Crash", "No crash")
    df = c[["odiNumber", "make", "model", "model_year", "segment",
            "component_primary", "summary", "label"]].reset_index(drop=True)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=df["label"])
    assert not (set(train["odiNumber"]) & set(test["odiNumber"])), "split not disjoint"
    # Prove this is the very same split the Naive Bayes tab used.
    nb_te = pd.read_csv(CLEAN / "nb_test_sample.csv")
    assert (test["odiNumber"].head(len(nb_te)).values == nb_te["odiNumber"].values).all(), \
        "split differs from the Naive Bayes tab"
    print(f"    labelled {len(df):,}   train {len(train):,}   test {len(test):,}  "
          f"(matches the Naive Bayes split)")
    train.head(400).to_csv(CLEAN / "reg_logistic_train_sample.csv", index=False)
    test.head(400).to_csv(CLEAN / "reg_logistic_test_sample.csv", index=False)
    return df, train, test


def logistic_cv(Xtr, ytr):
    """Two settings chosen by 5-fold stratified CV on the TRAINING set: the
    inverse regularisation strength C, and whether the rare class is
    reweighted. Scored on F1 for the Crash class, because accuracy on a 4.7%
    class rewards a model for predicting 'no crash' everywhere."""
    print("\n[B2] CROSS-VALIDATION ON THE TRAINING SET")
    Cs = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for cw in (None, "balanced"):
        for C in Cs:
            m = LogisticRegression(C=C, class_weight=cw, solver="liblinear", max_iter=2000)
            p = cross_val_predict(m, Xtr, ytr, cv=skf, n_jobs=-1)
            rows.append({"class_weight": cw or "none", "C": C,
                         "cv_accuracy": accuracy_score(ytr, p),
                         "cv_f1_crash": f1_score(ytr, p, pos_label="Crash"),
                         "cv_recall_crash": recall_score(ytr, p, pos_label="Crash"),
                         "cv_precision_crash": precision_score(ytr, p, pos_label="Crash")})
            print(f"    class_weight={cw or 'none':<9} C={C:<6g} "
                  f"acc={rows[-1]['cv_accuracy']:.4f} F1={rows[-1]['cv_f1_crash']:.4f}", flush=True)
    cv = pd.DataFrame(rows)
    best = cv.loc[cv["cv_f1_crash"].idxmax()]
    print(f"    -> chosen: C={best['C']:g}, class_weight={best['class_weight']} "
          f"(CV F1 {best['cv_f1_crash']:.4f})")
    cv.round(5).to_csv(CLEAN / "reg_logistic_cv.csv", index=False)
    return cv, best


def fig_logistic_cv(cv, best):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6), sharex=True)
    for i, (cw, g) in enumerate(cv.groupby("class_weight", sort=False)):
        kw = dict(color=SLOT[i], lw=2, marker="o", ms=5, mfc=SLOT[i], mec=SURFACE,
                  mew=1.3, label=f"class weight = {cw}")
        a1.plot(g["C"], g["cv_f1_crash"] * 100, **kw)
        a2.plot(g["C"], g["cv_accuracy"] * 100, **kw)
    for ax in (a1, a2):
        ax.axvline(best["C"], color=INK_SOFT, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("C, inverse regularisation strength (log scale)")
        ax.legend(loc="lower right", fontsize=9)
        frame(ax)
    a1.annotate(f"chosen: C = {best['C']:g}, {best['class_weight']}",
                xy=(best["C"], best["cv_f1_crash"] * 100),
                xytext=(-150, -145), textcoords="offset points", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1))
    a1.set_ylabel("F1 on the Crash class (%)")
    a1.set_title("Selection criterion: cross-validated F1", fontsize=11.5)
    a2.set_ylabel("Accuracy (%)")
    a2.set_title("Cross-validated accuracy, for reference", fontsize=11.5)
    fig.suptitle("Tuning logistic regression with 5-fold CV on the training set only",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    save(fig, "reg_fig_logistic_cv.png")


def fig_odds(coef_df, n=12):
    top = coef_df.nlargest(n, "coef")
    bot = coef_df.nsmallest(n, "coef")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    a1.barh(top["term"][::-1], top["odds_ratio"][::-1], color=SIGNAL, height=0.68)
    a1.set_title("Each extra use multiplies the odds of 'Crash' by...", fontsize=11.5)
    a1.set_xlabel("Odds ratio, exp(coefficient)")
    a1.axvline(1, color=INK_SOFT, lw=1)
    frame(a1, axis="x"); a1.grid(axis="x"); a1.grid(axis="y", visible=False)
    a2.barh(bot["term"], bot["odds_ratio"], color=SLOT[0], height=0.68)
    a2.set_title("...or divides them (odds ratio below 1)", fontsize=11.5)
    a2.set_xlabel("Odds ratio, exp(coefficient)")
    a2.axvline(1, color=INK_SOFT, lw=1)
    frame(a2, axis="x"); a2.grid(axis="x"); a2.grid(axis="y", visible=False)
    save(fig, "reg_fig_logistic_odds.png")


def fig_vs_nb(res):
    r = res[res["model"] != "Majority-class baseline"]
    metrics = ["accuracy", "precision_crash", "recall_crash", "f1_crash"]
    nice = ["Accuracy", "Precision\n(Crash)", "Recall\n(Crash)", "F1\n(Crash)"]
    x = np.arange(len(metrics))
    w = 0.8 / len(r)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for i, (_, row) in enumerate(r.iterrows()):
        ax.bar(x + (i - (len(r) - 1) / 2) * w, [row[m] * 100 for m in metrics], w,
               color=SLOT[i], label=row["model"], edgecolor=SURFACE, linewidth=1.5)
    base = res[res["model"] == "Majority-class baseline"]["accuracy"].iloc[0] * 100
    ax.axhline(base, color=INK_SOFT, lw=1.2)
    ax.text(len(metrics) - 0.5, base + 1.5, f"majority-class baseline accuracy {base:.1f}%",
            ha="right", fontsize=9.5, color=INK_SOFT)
    ax.set_xticks(x, nice)
    ax.set_ylabel("Percent, on the same 20,821 test complaints")
    ax.set_title("Logistic regression against the two Naive Bayes models")
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.set_ylim(0, 112)
    frame(ax)
    save(fig, "reg_fig_vs_nb.png")


# ======================================================================
# CONCEPT DIAGRAMS
# ======================================================================

def concept_linear_vs_logistic():
    rng = np.random.default_rng(21)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

    x = rng.uniform(0, 10, 40)
    y = 1.5 + 0.8 * x + rng.normal(0, 1.2, 40)
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(0, 10, 50)
    for xi, yi in zip(x, y):
        a1.plot([xi, xi], [yi, b0 + b1 * xi], color=GRID, lw=1.2, zorder=1)
    a1.scatter(x, y, s=38, color=SLOT[0], edgecolor=SURFACE, linewidth=0.8, zorder=3)
    a1.plot(xx, b0 + b1 * xx, color=SIGNAL, lw=2.4, zorder=2)
    a1.text(0.3, 9.6, f"y = {b0:.1f} + {b1:.2f}x", color=SIGNAL, fontsize=11)
    a1.text(6.2, 1.2, "grey lines = residuals,\nthe errors being squared\nand minimised",
            fontsize=9.5, color=INK_SOFT)
    a1.set_title("Linear regression predicts a number", fontsize=11.5)
    a1.set_xlabel("Input feature"); a1.set_ylabel("Numeric target")
    frame(a1, axis="both")

    xc = rng.uniform(0, 10, 60)
    pr = 1 / (1 + np.exp(-(xc - 5) * 1.3))
    yc = (rng.random(60) < pr).astype(int)
    a2.scatter(xc, yc + rng.normal(0, 0.02, 60), s=38,
               color=[SIGNAL if v else SLOT[0] for v in yc],
               edgecolor=SURFACE, linewidth=0.8, zorder=3)
    a2.plot(xx, 1 / (1 + np.exp(-(xx - 5) * 1.3)), color=INK, lw=2.4)
    a2.set_yticks([0, 0.5, 1], ["0  (No crash)", "0.5", "1  (Crash)"])
    a2.set_title("Logistic regression predicts a probability of a class", fontsize=11.5)
    a2.set_xlabel("Input feature"); a2.set_ylabel("P(class = 1)")
    frame(a2, axis="both")
    save(fig, "reg_concept_linear_vs_logistic.png")


def concept_sigmoid():
    z = np.linspace(-7, 7, 300)
    s = 1 / (1 + np.exp(-z))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    a1.plot(z, s, color=INK, lw=2.6)
    a1.axhline(0.5, color=SIGNAL, lw=1.3)
    a1.axvline(0, color=GRID, lw=1.2)
    a1.fill_between(z, 0, 1, where=z >= 0, color=SIGNAL, alpha=0.06)
    a1.fill_between(z, 0, 1, where=z < 0, color=SLOT[0], alpha=0.06)
    a1.text(3.4, 0.2, "probability ≥ 0.5\n→ predict Crash", color=SIGNAL,
            fontsize=10, ha="center")
    a1.text(-3.4, 0.72, "probability < 0.5\n→ predict No crash", color=SLOT[0],
            fontsize=10, ha="center")
    a1.text(-6.8, 0.53, "threshold 0.5", color=SIGNAL, fontsize=9.5)
    a1.set_xlabel("Linear score  z = b₀ + b₁x₁ + b₂x₂ + ...")
    a1.set_ylabel("Probability  σ(z) = 1 / (1 + e⁻ᶻ)")
    a1.set_title("The sigmoid squeezes any score into a probability", fontsize=11.5)
    frame(a1, axis="both")

    a2.axis("off")
    a2.set_xlim(0, 10); a2.set_ylim(0, 10)
    steps = [("1. Linear score", "z = b₀ + Σ bᵢxᵢ", "can be any number"),
             ("2. Sigmoid", "p = 1 / (1 + e⁻ᶻ)", "always between 0 and 1"),
             ("3. Threshold", "p ≥ 0.5 ?", "yes → Crash, no → No crash"),
             ("Odds reading", "odds = p / (1 − p) = eᶻ", "each bᵢ multiplies odds by e^bᵢ")]
    y = 8.6
    for i, (h, f, n) in enumerate(steps):
        fc = "#fdecef" if i == 2 else "#eef5fd"
        a2.add_patch(plt.Rectangle((0.4, y - 1.0), 9.2, 1.75, facecolor=fc,
                                   edgecolor=GRID, linewidth=1.2))
        a2.text(0.8, y + 0.25, h, fontsize=10.5, fontweight="bold", color=INK)
        a2.text(0.8, y - 0.5, f, fontsize=10.5, color=INK)
        a2.text(9.3, y - 0.5, n, fontsize=9, color=INK_SOFT, ha="right", style="italic")
        if i < 2:
            a2.annotate("", xy=(5, y - 1.15), xytext=(5, y - 0.95),
                        arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.6))
        y -= 2.3
    a2.set_title("From features to a class, in three steps", fontsize=11.5)
    save(fig, "reg_concept_sigmoid.png")


# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("REGRESSION  --  linear (crash rate) and logistic (crash / no crash)")
    print("=" * 72)

    # ---------------- linear ----------------
    d, SEG = linear_data()
    tr, te = linear_split(d)
    cv_lin, sets, best_lin, base_rmse = linear_cv(tr, SEG)
    chosen, ols, cols, preds, res_lin, coef_lin, lab_row, yte_lin = linear_final(tr, te, sets, best_lin)

    # ---------------- logistic ----------------
    df, train, test = logistic_data()
    vec = CountVectorizer(stop_words="english", min_df=5, max_features=6000, lowercase=True)
    Xtr = vec.fit_transform(train["summary"])        # fitted on TRAIN only
    Xte = vec.transform(test["summary"])
    ytr, yte = train["label"].values, test["label"].values
    print(f"    document-term matrix: train {Xtr.shape}, test {Xte.shape}")

    cv_log, best_log = logistic_cv(Xtr, ytr)
    cw = None if best_log["class_weight"] == "none" else "balanced"
    logit = LogisticRegression(C=float(best_log["C"]), class_weight=cw,
                               solver="liblinear", max_iter=2000).fit(Xtr, ytr)
    p_log = logit.predict(Xte)

    print("\n[B3] FINAL COMPARISON ON THE SAME TEST SET")
    mnb = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    bnb = BernoulliNB(alpha=1.0, binarize=0.5).fit(Xtr, ytr)
    p_mnb, p_bnb = mnb.predict(Xte), bnb.predict(Xte)
    p_maj = np.full(len(yte), pd.Series(ytr).value_counts().idxmax())
    res_log = pd.DataFrame([clf_metrics(f"Logistic regression (C={best_log['C']:g}, "
                                        f"class weight {best_log['class_weight']})", yte, p_log),
                            clf_metrics("Multinomial NB", yte, p_mnb),
                            clf_metrics("Bernoulli NB", yte, p_bnb),
                            clf_metrics("Majority-class baseline", yte, p_maj)]).round(4)
    print(res_log.to_string(index=False))
    nb_pub = pd.read_csv(CLEAN / "nb_results.csv").set_index("model")
    for m in ("Multinomial NB", "Bernoulli NB"):
        got = res_log.set_index("model").loc[m, "f1_crash"]
        assert abs(got - nb_pub.loc[m, "f1_crash"]) < 1e-3, f"{m} does not reproduce Part 3"
    print("    Naive Bayes figures reproduce the published Part 3 results exactly")
    res_log.to_csv(CLEAN / "reg_logistic_results.csv", index=False)

    terms = np.array(vec.get_feature_names_out())
    pos = list(logit.classes_).index("Crash")
    sign = 1 if pos == 1 else -1     # coef_ is for classes_[1]
    coef_df = pd.DataFrame({"term": terms, "coef": sign * logit.coef_[0]})
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df = coef_df.sort_values("coef", ascending=False)
    coef_df.round(5).to_csv(CLEAN / "reg_logistic_coefficients.csv", index=False)
    print("\n    strongest crash terms:", ", ".join(
        f"{t} x{o:.2f}" for t, o in coef_df.head(8)[["term", "odds_ratio"]].values))
    print("    strongest no-crash terms:", ", ".join(
        f"{t} x{o:.2f}" for t, o in coef_df.tail(8)[["term", "odds_ratio"]].values))
    print(f"    intercept (log-odds with an empty narrative): "
          f"{sign * logit.intercept_[0]:.3f}")

    # ---------------- figures ----------------
    print("\n[C] FIGURES")
    concept_linear_vs_logistic()
    concept_sigmoid()

    show = ["make", "model", "model_year", "segment", "OverallRating",
            "rollover_prob_pct", "has_fcw", "mean_vehicle_age", "log_complaints",
            "crash_rate_pct"]
    table_png(d[show], f"Linear regression data: {len(d)} model-years, numeric target "
              "crash_rate_pct in the last column", "reg_lin_sample.png", width=15)
    table_png(tr[show], f"TRAINING set: {len(tr)} model-years ({1-TEST_SIZE:.0%})",
              "reg_lin_train.png", width=15)
    table_png(te[show], f"TESTING set: {len(te)} model-years ({TEST_SIZE:.0%}), never "
              "used to fit or tune", "reg_lin_test.png", width=15)
    table_png(df[["odiNumber", "make", "segment", "summary", "label"]],
              f"Logistic regression data: {len(df):,} complaints, two-class label in the "
              "last column", "reg_log_sample.png", width=14)
    table_png(train[["odiNumber", "make", "segment", "summary", "label"]],
              f"TRAINING set: {len(train):,} complaints ({1-TEST_SIZE:.0%}), also used "
              "for cross-validation", "reg_log_train.png", width=14)
    table_png(test[["odiNumber", "make", "segment", "summary", "label"]],
              f"TESTING set: {len(test):,} complaints ({TEST_SIZE:.0%}), the same rows "
              "Naive Bayes was tested on", "reg_log_test.png", width=14)

    fig_linear_cv(cv_lin, base_rmse)
    best_name = res_lin.iloc[0]["model"]
    fig_pred_vs_actual(yte_lin, preds[best_name], res_lin.iloc[0]["test_r2"],
                       res_lin.iloc[0]["test_rmse"])
    fig_coefficients(coef_lin)
    fig_logistic_cv(cv_log, best_log)
    fig_confusion(yte, p_log, "Logistic regression", "reg_fig_logistic_confusion.png")
    fig_odds(coef_df)
    fig_vs_nb(res_log)
    print("done.")
