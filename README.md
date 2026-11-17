# Rated Safe, Reported Broken

**CSCI 5612 — Machine Learning · Semester Project**
What crash-test scores miss about how vehicles actually fail.

**Live site:** https://Rajeshwar987.github.io/csci5612-project/

---

## The question

A vehicle's crash-test star rating is earned in a laboratory, in a controlled
collision lasting a fraction of a second. The fifteen years of driving that
follow are not controlled at all. This project asks whether a strong crash-test
score predicts a quiet ownership experience, or whether the two are largely
independent dimensions of safety.

## Data sources

All data are published by the U.S. National Highway Traffic Safety
Administration and are in the public domain. Two independent gathering routes
are used.

**APIs** (keyless, queried with Python `requests`):

| Source | Endpoint |
|---|---|
| Complaints | `api.nhtsa.gov/complaints/complaintsByVehicle` |
| Recalls | `api.nhtsa.gov/recalls/recallsByVehicle` |
| Safety Ratings | `api.nhtsa.gov/SafetyRatings/...` |

Example request:

```
GET https://api.nhtsa.gov/complaints/complaintsByVehicle?make=honda&model=accord&modelYear=2020
```

**Bulk download:** `https://static.nhtsa.gov/odi/ffdd/inv/FLAT_INV.zip`
(Office of Defects Investigation flat file.)

Coverage: 39 vehicle models across 16 makes, model years 2016–2024.

## What is here

| Table | Source | Raw rows | Clean rows |
|---|---|---:|---:|
| Complaints | API | 84,135 | 83,967 |
| Recalls | API | 1,662 | 1,662 |
| Safety ratings | API | 505 | 421 |
| Investigations | Bulk download | 154,380 | 138,271 |
| Joined model-year table | Derived | — | 229 |

## Repository layout

```
code/
  fetch_data.py               gather from 3 APIs + 1 bulk download (cached, resumable)
  clean_data.py               clean, repair, derive, join -> data/clean/
  eda_visualize.py            EDA figures and raw/clean snapshots -> figures/
  clustering_analysis.py      k-means and cosine hierarchical clustering
  pca_analysis.py             principal component analysis
  sample_snapshots.py         data-prep images for clustering and PCA
  naive_bayes_analysis.py     Multinomial and Bernoulli Naive Bayes
  decision_tree_analysis.py   three decision trees
  regression_analysis.py      linear and logistic regression
  adaboost_analysis.py        AdaBoost with decision stumps
  build_site.py               generate the website
notebooks/                    one annotated notebook per method, with outputs
data/
  raw/                        gathered data, gzipped, exactly as received
  clean/                      cleaned tables, model inputs, samples and results
figures/                      every figure on the site
assets/                       stylesheet and images
*.html                        the website
```

## Reproducing

```bash
pip install -r requirements.txt
py code/fetch_data.py       # ~10 min cold; instant from cache
py code/clean_data.py
py code/eda_visualize.py
py code/clustering_analysis.py
py code/pca_analysis.py
py code/sample_snapshots.py
py code/naive_bayes_analysis.py
py code/decision_tree_analysis.py
py code/regression_analysis.py
py code/adaboost_analysis.py
py code/build_site.py
```

Every analysis uses the fixed seed 5612. Supervised settings are chosen on
validation data or by cross-validation inside the training set; each test set
is used once, to score the final model.

API responses are cached under `data/raw/cache/` (gitignored, ~86 MB) so the
gathering step is resumable and does not repeatedly load the public service.
Every dataset derived from that cache is committed.

## Notes on the data

Findings on this site are associations in observational data, not causal
results. The complaint record is self-selected: it reflects what owners chose
to report, not everything that went wrong. Vehicle identification numbers are
truncated to the three-character manufacturer prefix; no personal data is
stored in this repository.

## Project modules

| Module | Deliverable | Status |
|---|---|---|
| 1 | Topic, data gathering, cleaning, EDA | Complete |
| 2 | Clustering and PCA | Complete |
| 3 | Naive Bayes and Decision Trees | Complete |
| 4 | Regression (linear and logistic) and AdaBoost | Complete |
