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
ASSIGNMENT = "Module 1 Project Part 1"
URL = "https://Rajeshwar987.github.io/csci5612-project/"

OUT = Path.home() / "Downloads" / f"{NAME} Project A1.docx"

doc = Document()

h = doc.add_heading(f"{ASSIGNMENT} - Project Website", level=1)

p = doc.add_paragraph()
p.add_run(f"{NAME}\n").bold = True
p.add_run(f"{COURSE}\n")
p.add_run(ASSIGNMENT)

doc.add_paragraph()

lead = doc.add_paragraph()
lead.add_run("Project website URL:").bold = True

link = doc.add_paragraph()
run = link.add_run(URL)
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(0x0B, 0x51, 0x94)
run.underline = True

doc.add_paragraph()

doc.add_paragraph(
    "Project topic: whether laboratory crash-test ratings predict the safety "
    "problems vehicle owners actually report once a vehicle is on the road."
)

doc.add_paragraph(
    "The site contains the Introduction tab (written paragraphs, an image, and "
    "the ten research questions) and the DataPrep_EDA tab (data gathering via "
    "three NHTSA APIs and one bulk download, the full cleaning record, raw and "
    "cleaned data snapshots, and sixteen exploratory visualisations). Tabs for "
    "Clustering, PCA, NaiveBayes, DecTrees, SVMs, Regression, NN, Conclusions "
    "and About Me are in place for the later modules."
)

doc.add_paragraph(
    "All code and all raw and cleaned data are linked from the website and are "
    "public at https://github.com/Rajeshwar987/csci5612-project"
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
