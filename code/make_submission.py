"""
make_submission.py  --  builds the Canvas submission document.

The assignment asks for a single URL placed in a .docx file, with the document
named for the student and the assignment.
"""

from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

NAME = "Rajeshwar Rengeswaran"
COURSE = "CSCI 5612 - Machine Learning"
ASSIGNMENT = "Module 2 Project Part 2"
URL = "https://rajeshwar987.github.io/csci5612-project/"
REPO = "https://github.com/Rajeshwar987/csci5612-project"

OUT = Path.home() / "Downloads" / f"{NAME} Project A2.docx"

doc = Document()

h = doc.add_heading(f"{ASSIGNMENT} - Project Website", level=1)

p = doc.add_paragraph()
p.add_run(f"{NAME}\n").bold = True
p.add_run(f"{COURSE}\n")
p.add_run(ASSIGNMENT)

doc.add_paragraph()

def link_line(label, url):
    p = doc.add_paragraph()
    p.add_run(f"{label}\n").bold = True
    run = p.add_run(url)
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x0B, 0x51, 0x94)
    run.underline = True


link_line("Project website URL:", URL)
link_line("GitHub repository:", REPO)

doc.add_paragraph()

doc.add_paragraph(
    "Project topic: whether laboratory crash-test ratings predict the safety "
    "problems vehicle owners actually report once a vehicle is on the road."
)

doc.add_heading("Direct links", level=2)

for label, url in [
    ("Clustering tab", f"{URL}clustering.html"),
    ("PCA tab", f"{URL}pca.html"),
    ("Introduction tab", URL),
    ("Clustering notebook (.ipynb, with outputs)",
     f"{REPO}/blob/main/notebooks/clustering.ipynb"),
    ("PCA notebook (.ipynb, with outputs)",
     f"{REPO}/blob/main/notebooks/pca.ipynb"),
    ("Clustering script", f"{REPO}/blob/main/code/clustering_analysis.py"),
    ("PCA script", f"{REPO}/blob/main/code/pca_analysis.py"),
    ("Cleaned data folder", f"{REPO}/tree/main/data/clean"),
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(f"{label}: ")
    r = p.add_run(url)
    r.font.color.rgb = RGBColor(0x0B, 0x51, 0x94)
    r.underline = True

doc.add_heading("What is new in Part 2", level=2)

for line in [
    "Introduction: now five body paragraphs of at least ten sentences each, "
    "with two images and the ten research questions retained.",
    "Clustering tab: Overview (partitional versus hierarchical, Euclidean "
    "versus cosine distance, two explanatory images, and how clustering is "
    "used in this project), Data Prep (before and after data frame images, "
    "unlabeled numeric requirement explained), Code (linked, not pasted), "
    "Results and Conclusions.",
    "k-means was swept from k=2 to k=15 and compared at k=3, 5 and 7. "
    "Hierarchical clustering used cosine distance with average linkage, since "
    "Ward linkage is undefined for cosine. Both methods independently select "
    "k=7, with an adjusted Rand index of 0.638.",
    "PCA tab: Overview (eigenvalues and eigenvectors, dimensionality "
    "reduction, the challenges of high-dimensional data, two explanatory "
    "images, and how PCA is used in this project), Data Prep with before and "
    "after images, Code, Results and Conclusions. PC1 carries 24.2% of the "
    "variance and nine of sixteen components are needed to reach 90%.",
    "Twenty-two figures in total across the two tabs, including silhouette "
    "curve and per-vehicle widths, a three-way k comparison, full and "
    "truncated dendrograms, a method agreement matrix, cluster profiles, "
    "scree and eigenvalue plots, a loadings matrix, a correlation circle and "
    "a cluster projection.",
    "Both Conclusions sections are written non-technically and tied to the "
    "project topic rather than to the methods.",
]:
    doc.add_paragraph(line, style="List Bullet")

doc.add_paragraph()
doc.add_paragraph(
    "All analysis code, notebooks, and all raw, cleaned and intermediate data "
    "are linked from the relevant tabs and are public in the repository above. "
    "No code is pasted as an image anywhere on the site."
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
