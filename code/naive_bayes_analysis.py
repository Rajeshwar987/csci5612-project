"""
naive_bayes_analysis.py  --  CSCI 5612 Project, Module 3 Part 3
Supervised classification of complaint narratives with Naive Bayes.

Task: given only the words an owner wrote when filing a safety complaint,
predict whether that complaint involved a crash.

Multinomial NB expects non-negative integer features, which is exactly what a
word-count matrix is, so the text is vectorised with CountVectorizer. Bernoulli
NB is fitted alongside it on the same split for comparison, using presence /
absence rather than counts.

Outputs:
    data/clean/nb_labeled_sample.csv      the labelled data before splitting
    data/clean/nb_train_sample.csv        a sample of the training set
    data/clean/nb_test_sample.csv         a sample of the testing set
    data/clean/nb_results.csv             metrics for every model fitted
    figures/nb_*.png                      concept diagrams, results, matrices
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

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
        sub[c] = sub[c].astype(str).str.slice(0, 44)
    fig, ax = plt.subplots(figsize=(width, 0.42 * nrows + 1.4))
    ax.axis("off")
    t = ax.table(cellText=sub.values,
                 colLabels=[c.replace("_", " ")[:20] for c in sub.columns],
                 cellLoc="left", loc="center")
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


def load_labeled():
    print("\n[1] LABELLED DATA")
    c = pd.read_csv(CLEAN / "complaints_clean.csv.gz", low_memory=False)
    c["summary"] = c["summary"].fillna("").astype(str)

    # A narrative with almost no words cannot support a text model.
    before = len(c)
    c = c[c["summary"].str.split().str.len() >= 5]
    print(f"    dropped narratives under 5 words: {before:,} -> {len(c):,}")

    # The label. This is what makes the task supervised: every row carries a
    # known, recorded answer that the model is trained to reproduce.
    c["label"] = np.where(c["crash"], "Crash", "No crash")

    df = c[["odiNumber", "make", "model", "model_year", "segment",
            "component_primary", "summary", "label"]].reset_index(drop=True)
    print(f"    labelled rows: {len(df):,}")
    print(f"    class balance: {df['label'].value_counts(normalize=True).round(4).to_dict()}")
    df.to_csv(CLEAN / "nb_labeled_sample.csv", index=False)
    return df



def split(df):
    print("\n[2] TRAIN / TEST SPLIT")
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df["label"])          # keep the rare class proportional

    overlap = set(train["odiNumber"]) & set(test["odiNumber"])
    print(f"    train: {len(train):,}   test: {len(test):,}")
    print(f"    overlapping complaint IDs between the two sets: {len(overlap)}")
    assert not overlap, "train and test are not disjoint"
    print(f"    train balance: {train['label'].value_counts(normalize=True).round(4).to_dict()}")
    print(f"    test  balance: {test['label'].value_counts(normalize=True).round(4).to_dict()}")

    train.head(400).to_csv(CLEAN / "nb_train_sample.csv", index=False)
    test.head(400).to_csv(CLEAN / "nb_test_sample.csv", index=False)
    return train, test



def fit_models(train, test):
    print("\n[3] VECTORISING AND FITTING")
    vec = CountVectorizer(stop_words="english", min_df=5, max_features=6000,
                          lowercase=True)
    Xtr = vec.fit_transform(train["summary"])    # fitted on TRAIN ONLY
    Xte = vec.transform(test["summary"])         # test is only transformed
    ytr, yte = train["label"].values, test["label"].values
    print(f"    vocabulary learned from the training set only: {len(vec.vocabulary_):,} terms")
    print(f"    train matrix {Xtr.shape}, test matrix {Xte.shape}")
    print(f"    matrix is non-negative integer counts: "
          f"min={Xtr.min()}, dtype={Xtr.dtype}")

    results = []

    mnb = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    pm = mnb.predict(Xte)

    # Bernoulli NB uses presence/absence rather than how many times a word
    # appeared, so the same matrix is binarised for it.
    bnb = BernoulliNB(alpha=1.0, binarize=0.5).fit(Xtr, ytr)
    pb = bnb.predict(Xte)

    majority = pd.Series(ytr).value_counts().idxmax()
    pmaj = np.full(len(yte), majority)

    for name, pred in (("Multinomial NB", pm), ("Bernoulli NB", pb),
                       ("Majority-class baseline", pmaj)):
        acc = accuracy_score(yte, pred)
        results.append({
            "model": name,
            "accuracy": acc,
            "precision_crash": precision_score(yte, pred, pos_label="Crash",
                                               zero_division=0),
            "recall_crash": recall_score(yte, pred, pos_label="Crash",
                                         zero_division=0),
            "f1_crash": f1_score(yte, pred, pos_label="Crash", zero_division=0),
        })
        print(f"    {name:<26} accuracy={acc:.4f}")

    res = pd.DataFrame(results).round(4)
    print("\n" + res.to_string(index=False))
    print("\n  Multinomial NB, full report:")
    print(classification_report(yte, pm, zero_division=0))
    return vec, mnb, bnb, Xtr, Xte, ytr, yte, pm, pb, res



def smoothing_sweep(Xtr, ytr, Xte, yte):
    """Why smoothing is required, measured rather than asserted."""
    print("\n[4] SMOOTHING SWEEP")
    alphas = [1e-10, 1e-4, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    rows = []
    for a in alphas:
        m = MultinomialNB(alpha=a).fit(Xtr, ytr)
        p = m.predict(Xte)
        rows.append({"alpha": a,
                     "accuracy": accuracy_score(yte, p),
                     "f1_crash": f1_score(yte, p, pos_label="Crash", zero_division=0)})
        print(f"    alpha={a:<8g} accuracy={rows[-1]['accuracy']:.4f} "
              f"f1(Crash)={rows[-1]['f1_crash']:.4f}")
    return pd.DataFrame(rows)


def fig_confusion(yte, pred, title, fname, acc):
    labels = ["No crash", "Crash"]
    cm = confusion_matrix(yte, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap=SEQ)
    ax.set_xticks([0, 1], [f"Predicted\n{l}" for l in labels])
    ax.set_yticks([0, 1], [f"Actual\n{l}" for l in labels])
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}\n({v/total:.1%})", ha="center", va="center",
                    fontsize=13, color=SURFACE if v > cm.max() * 0.5 else INK)
    ax.set_title(f"{title}\naccuracy {acc:.1%}", fontsize=12.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, fname)


def fig_smoothing(sw):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(sw["alpha"], sw["accuracy"] * 100, color=SLOT[0], lw=2, marker="o",
            ms=5, mfc=SLOT[0], mec=SURFACE, mew=1.4, label="Overall accuracy")
    ax.plot(sw["alpha"], sw["f1_crash"] * 100, color=SIGNAL, lw=2, marker="s",
            ms=5, mfc=SIGNAL, mec=SURFACE, mew=1.4, label="F1 on the Crash class")
    ax.set_xscale("log")
    best = sw.loc[sw["f1_crash"].idxmax()]
    ax.annotate(f"best F1 at alpha={best['alpha']:g}",
                xy=(best["alpha"], best["f1_crash"] * 100),
                xytext=(10, 14), textcoords="offset points", fontsize=10, color=INK)
    ax.set_title("Effect of Laplace smoothing on Multinomial Naive Bayes")
    ax.set_xlabel("Smoothing parameter alpha (log scale)")
    ax.set_ylabel("Percent")
    ax.legend(loc="center left")
    frame(ax)
    save(fig, "nb_fig_smoothing.png")


def fig_top_terms(vec, mnb, n=14):
    """Which words move the model toward each class."""
    terms = np.array(vec.get_feature_names_out())
    classes = list(mnb.classes_)
    i_crash = classes.index("Crash")
    i_no = classes.index("No crash")
    llr = mnb.feature_log_prob_[i_crash] - mnb.feature_log_prob_[i_no]
    top_crash = np.argsort(llr)[-n:]
    top_no = np.argsort(llr)[:n]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    a1.barh(terms[top_crash], llr[top_crash], color=SIGNAL, height=0.68)
    a1.set_title("Words pushing a complaint toward 'Crash'", fontsize=12)
    a1.set_xlabel("Log-probability ratio")
    frame(a1, axis="x"); a1.grid(axis="x"); a1.grid(axis="y", visible=False)

    a2.barh(terms[top_no], llr[top_no], color=SLOT[0], height=0.68)
    a2.set_title("Words pushing it toward 'No crash'", fontsize=12)
    a2.set_xlabel("Log-probability ratio")
    frame(a2, axis="x"); a2.grid(axis="x"); a2.grid(axis="y", visible=False)
    save(fig, "nb_fig_top_terms.png")
    return terms[top_crash][::-1], terms[top_no]


def fig_model_compare(res):
    r = res[res["model"] != "Majority-class baseline"]
    metrics = ["accuracy", "precision_crash", "recall_crash", "f1_crash"]
    nice = ["Accuracy", "Precision\n(Crash)", "Recall\n(Crash)", "F1\n(Crash)"]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (_, row) in enumerate(r.iterrows()):
        ax.bar(x + (i - 0.5) * w, [row[m] * 100 for m in metrics], w,
               color=SLOT[i], label=row["model"], edgecolor=SURFACE, linewidth=2)
    base = res[res["model"] == "Majority-class baseline"]["accuracy"].iloc[0] * 100
    ax.axhline(base, color=INK_SOFT, lw=1.2)
    ax.text(len(metrics) - 0.5, base + 1.5,
            f"majority-class baseline accuracy {base:.1f}%",
            ha="right", fontsize=9.5, color=INK_SOFT)
    ax.set_xticks(x, nice)
    ax.set_ylabel("Percent")
    ax.set_title("Multinomial and Bernoulli Naive Bayes compared on the same split")
    ax.legend(loc="upper right")
    frame(ax)
    save(fig, "nb_fig_model_compare.png")


# ----- concept diagrams -------------------------------------------------

def concept_how_nb_works():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    def box(x, y, w, h, text, fc, tc=INK, fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc,
                                   edgecolor=GRID, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc)

    box(0.2, 3.1, 2.3, 1.2, "A complaint\n\"brakes failed and\nI hit a pole\"", "#f5f7fa")
    box(0.2, 0.7, 2.3, 1.2, "Count the words\nbrakes:1 failed:1\nhit:1 pole:1", "#f5f7fa")
    ax.annotate("", xy=(1.35, 1.95), xytext=(1.35, 3.05),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))

    # Spans the full height so both incoming arrows land on the box.
    box(3.1, 1.2, 2.9, 3.1,
        "For each class, multiply\nthe chance of seeing\neach word in that class\n"
        r"$P(c)\prod_i P(w_i \mid c)$", "#eef5fd", fs=10)
    ax.annotate("", xy=(3.05, 1.3), xytext=(2.55, 1.3),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
    ax.annotate("", xy=(3.05, 3.1), xytext=(2.55, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))

    box(6.6, 3.3, 3.1, 0.95, "P(Crash | words)  =  0.87", "#fdecef", SIGNAL, 11)
    box(6.6, 2.1, 3.1, 0.95, "P(No crash | words)  =  0.13", "#f5f7fa", INK_SOFT, 11)
    ax.annotate("", xy=(6.55, 3.1), xytext=(6.05, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
    box(6.6, 0.7, 3.1, 1.0, "Predict the higher one:\nCRASH", "#fdecef", SIGNAL, 11)
    ax.annotate("", xy=(8.15, 1.75), xytext=(8.15, 2.05),
                arrowprops=dict(arrowstyle="-|>", color=SIGNAL, lw=1.8))

    ax.text(5, 4.72, "How Multinomial Naive Bayes classifies one complaint",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(5, 0.18, "\"Naive\" because it assumes each word is independent of the others "
                     "given the class - which is false, but works well in practice.",
            ha="center", fontsize=9.5, color=INK_SOFT, style="italic")
    save(fig, "nb_concept_how_it_works.png")


def concept_smoothing():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    words = ["brakes", "pedal", "collision", "airbag", "rollover"]
    counts_train = [42, 17, 9, 5, 0]
    colors = [SLOT[0]] * 4 + [SIGNAL]
    a1.bar(words, counts_train, color=colors, width=0.62)
    a1.set_title("The problem: a word never seen in a class", fontsize=11.5)
    a1.set_ylabel("Times seen in training, class 'Crash'")
    a1.text(4, 1.2, "count = 0\nso P(word|class) = 0\nand the whole product\ncollapses to 0",
            ha="center", fontsize=9.5, color=SIGNAL)
    frame(a1)

    alphas = np.array([0, 0.25, 0.5, 1.0])
    probs = (0 + alphas) / (1000 + alphas * 5000)
    a2.bar([f"alpha={a:g}" for a in alphas], probs * 1e4, color=
           [SIGNAL] + [SLOT[2]] * 3, width=0.6)
    a2.set_title("The fix: add alpha to every count", fontsize=11.5)
    a2.set_ylabel(r"P(word | class)  $\times 10^{4}$")
    a2.text(0, 0.06, "zero", ha="center", fontsize=9.5, color=SIGNAL)
    a2.text(1.9, max(probs * 1e4) * 0.78,
            "any alpha above 0 keeps\nthe probability non-zero,\nso one unseen word can no\n"
            "longer veto a prediction",
            fontsize=9.5, color=INK_SOFT)
    frame(a2)
    save(fig, "nb_concept_smoothing.png")


def concept_variants():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    words = ["brake", "fail", "crash", "warn", "smoke"]
    counts = [3, 2, 1, 0, 0]
    binary = [1, 1, 1, 0, 0]

    a1.bar(words, counts, color=SLOT[0], width=0.6)
    a1.set_title("Multinomial NB\nuses how many times a word appears", fontsize=11.5)
    a1.set_ylabel("Count in the document")
    a1.set_ylim(0, 3.6)
    for i, v in enumerate(counts):
        a1.text(i, v + 0.1, str(v), ha="center", fontsize=10, color=INK)
    frame(a1)

    a2.bar(words, binary, color=SLOT[3], width=0.6)
    a2.set_title("Bernoulli NB\nuses only whether it appears at all", fontsize=11.5)
    a2.set_ylabel("Present (1) or absent (0)")
    a2.set_ylim(0, 3.6)
    for i, v in enumerate(binary):
        a2.text(i, v + 0.1, str(v), ha="center", fontsize=10, color=INK)
    a2.text(3.5, 2.4, "Bernoulli also actively uses\nthe absent words as evidence,\n"
                      "which Multinomial ignores",
            ha="center", fontsize=9.5, color=INK_SOFT)
    frame(a2)
    save(fig, "nb_concept_variants.png")



if __name__ == "__main__":
    print("=" * 72)
    print("NAIVE BAYES  --  predicting a crash from the complaint narrative")
    print("=" * 72)

    df = load_labeled()
    train, test = split(df)
    vec, mnb, bnb, Xtr, Xte, ytr, yte, pm, pb, res = fit_models(train, test)
    sw = smoothing_sweep(Xtr, ytr, Xte, yte)

    res.to_csv(CLEAN / "nb_results.csv", index=False)
    sw.round(5).to_csv(CLEAN / "nb_smoothing_sweep.csv", index=False)

    print("\n[5] FIGURES")
    concept_how_nb_works()
    concept_smoothing()
    concept_variants()

    table_png(df[["odiNumber", "make", "model", "segment",
                  "component_primary", "summary", "label"]],
              f"Labelled data: {len(df):,} complaints, each with a known answer "
              "in the final column",
              "nb_sample_labeled.png")
    table_png(train[["odiNumber", "make", "segment", "summary", "label"]],
              f"TRAINING set: {len(train):,} rows ({1-TEST_SIZE:.0%}) - the model learns from these",
              "nb_sample_train.png", width=12)
    table_png(test[["odiNumber", "make", "segment", "summary", "label"]],
              f"TESTING set: {len(test):,} rows ({TEST_SIZE:.0%}) - held back, never seen during training",
              "nb_sample_test.png", width=12)

    fig_confusion(yte, pm, "Multinomial Naive Bayes", "nb_fig_confusion_mnb.png",
                  accuracy_score(yte, pm))
    fig_confusion(yte, pb, "Bernoulli Naive Bayes", "nb_fig_confusion_bnb.png",
                  accuracy_score(yte, pb))
    fig_smoothing(sw)
    tc, tn = fig_top_terms(vec, mnb)
    fig_model_compare(res)

    print(f"\n  top 'Crash' terms   : {', '.join(tc[:10])}")
    print(f"  top 'No crash' terms: {', '.join(tn[:10])}")
    print("\ndone.")
