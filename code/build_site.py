"""
build_site.py  --  CSCI 5612 Project, Module 1 Part 1
Generates the project website. Navigation lives in one place so every tab
stays consistent as later modules fill the remaining sections in.

Run:  py code/build_site.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/Rajeshwar987/csci5612-project"
BLOB = f"{REPO}/blob/main"
RAW_GH = f"{REPO}/raw/main"

TABS = [
    ("index.html", "Introduction"),
    ("dataprep_eda.html", "DataPrep_EDA"),
    ("clustering.html", "Clustering"),
    ("pca.html", "PCA"),
    ("naivebayes.html", "NaiveBayes"),
    ("dectrees.html", "DecTrees"),
    ("svms.html", "SVMs"),
    ("regression.html", "Regression"),
    ("nn.html", "NN"),
    ("conclusions.html", "Conclusions"),
    ("about.html", "About Me"),
]

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | Vehicle Safety Project</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
<header class="masthead">
  <div class="masthead-inner">
    <p class="course">CSCI 5612 &middot; Machine Learning &middot; Project</p>
    <h1>Rated Safe, Reported Broken</h1>
    <p class="byline">What crash-test scores miss about how vehicles actually fail</p>
    <nav class="tabs">
{nav}
    </nav>
  </div>
</header>
<main>
{body}
</main>
<footer><div>
  CSCI 5612 project website &middot; Rajeshwar Rengeswaran &middot;
  Data: <a href="https://www.nhtsa.gov/">U.S. National Highway Traffic Safety Administration</a> &middot;
  <a href="{repo}">Project code and data repository</a>
</div></footer>
</body>
</html>
"""


def nav_html(current):
    out = []
    for href, label in TABS:
        cls = ' class="active"' if href == current else ""
        out.append(f'      <a href="{href}"{cls}>{label}</a>')
    return "\n".join(out)


def figure(src, title, caption):
    """A figure with a title, and two sentences describing what it shows."""
    return f"""<figure>
  <img src="figures/{src}" alt="{title}">
  <figcaption><b>{title}.</b> {caption}</figcaption>
</figure>"""


def stub(method, module, overview, plan):
    """Placeholder tab carrying the required Overview / Data / Code / Results skeleton."""
    return f"""<h2>{method}</h2>
<p class="lede">{overview}</p>
<div class="callout">
  <p><b>This section is completed in Module {module}.</b> The structure below is in
  place and will be filled with the analysis, results, and visualisations for
  {method} as the course reaches that module.</p>
</div>

<h3>Overview</h3>
<p>{plan}</p>

<h3>Data</h3>
<p>The prepared data for this method will be drawn from the cleaned tables produced
in Module 1, and the exact subset, encoding, and normalisation used here will be
shown as a small sample image alongside links to both the raw and the cleaned files.</p>
<ul>
  <li>Raw data: <a href="{BLOB}/data/raw">data/raw/</a></li>
  <li>Cleaned data: <a href="{BLOB}/data/clean">data/clean/</a></li>
</ul>

<h3>Code</h3>
<p>The code for this section will be linked here once written. All project code is
Python, using pandas and scikit-learn, and lives in
<a href="{BLOB}/code">the repository's code folder</a>.</p>

<h3>Results</h3>
<div class="placeholder">
  <p>Results, parameter choices, accuracy measures, and visualisations for {method}
  will appear in this area, together with a discussion of what they mean for the
  question of how well crash-test ratings predict real-world vehicle problems.</p>
</div>"""


# ----------------------------------------------------------------------
# INTRODUCTION
# ----------------------------------------------------------------------

INTRO = f"""<h2>Introduction</h2>

<p class="lede">A new vehicle arrives with a safety score printed on its window
sticker. That number is earned in a laboratory, in a controlled collision that
lasts a fraction of a second. The years of driving that follow are not
controlled at all.</p>

<p>The automobile is the most dangerous machine most people operate, and it is
operated daily, casually, and usually without incident. Tens of thousands of
people die on American roads every year, and many times that number are injured
seriously enough to require hospital care. That toll is not distributed evenly:
it falls hardest on younger drivers, on rural roads, on people travelling at
night, and on those in older vehicles that never carried modern protective
equipment. Over the past six decades the response to this toll has been steady
regulatory pressure, and it has worked. Seat belts, crumple zones, airbags,
anti-lock braking, and electronic stability control each moved from optional
novelty to legal requirement, and each one measurably reduced the chance of
dying in a collision. Government crash testing sits at the centre of that
history, because a public score creates commercial pressure that regulation
alone cannot. Manufacturers compete on safety ratings in their advertising,
which means the rating itself has become a safety intervention. The question
that follows is how completely a laboratory score can describe the safety of a
machine that will be driven for fifteen years. A crash test measures what
happens during an impact. It does not measure what happens in the long,
uneventful period before one.</p>

<figure>
  <img src="assets/img/crashtest_side.jpg"
       alt="A silver sport-utility vehicle undergoing a laboratory side-impact crash test">
  <figcaption><b>A side-impact test in progress.</b> A moving barrier strikes a
  stationary vehicle at a fixed speed and angle while instrumented dummies record
  the forces reaching the occupants. Photograph by the U.S. National Highway
  Traffic Safety Administration, a public-domain federal work.</figcaption>
</figure>

<p>Safety is communicated to buyers as a small number of stars, and that
simplicity is deliberate. A shopper comparing vehicles in a showroom cannot be
expected to read engineering reports, so a five-point scale compresses an
enormous amount of testing into something glanceable. The tests behind the score
are rigorous and repeatable: vehicles are driven into barriers at prescribed
speeds, struck from the side by weighted sleds, and assessed for their tendency
to roll over during sharp manoeuvres. Sensors inside crash dummies measure the
loads reaching a head, a neck, a chest, and a femur, and those measurements
become the rating. What the rating describes, precisely, is how well a vehicle
protects a correctly belted adult during one specific kind of severe, sudden
event. It is silent on everything else. It says nothing about whether a fuel
pump will fail on a motorway three years later, whether a reversing camera will
go dark, whether a battery pack will ignite in a car park, or whether a steering
component will develop play at eighty thousand miles. Those failures are not
collisions, but they are unmistakably safety events, and some of them cause
collisions. A vehicle can therefore hold the highest possible laboratory score
and still accumulate a long record of dangerous faults in ordinary service.</p>

<p>There is a second, much messier record of how vehicles behave, and it is
written by the people who own them. When something goes wrong, an owner can file
a complaint with the federal safety regulator describing what happened, what
part failed, and whether anyone was hurt. These accounts arrive in their own
words, unevenly, sometimes within hours of an incident and sometimes years
afterwards. Individually they are anecdotes. Collected across hundreds of
thousands of vehicles they become an early-warning system, because a cluster of
similar complaints about the same component in the same model is how a
systematic defect first becomes visible. Regulators watch for those clusters,
open formal investigations into the ones that look real, and can compel a
manufacturer to issue a recall. That chain — complaint, investigation, recall —
is slow, adversarial, and public, and it is the mechanism by which a fault
affecting one owner eventually protects every other owner of that vehicle. It
also produces a record that is fundamentally different in character from a crash
test: retrospective rather than predictive, voluntary rather than standardised,
and driven by whatever owners happened to notice and bother to report.</p>

<p>These two records describe the same vehicles and almost never meet. One is
generated before a car is sold, under laboratory conditions, and is used to
market it. The other accumulates afterwards, in driveways and repair shops, and
surfaces mainly when something has already gone wrong. Placing them side by side
raises a question with real consequences for anyone choosing a car: does a strong
crash-test score predict a quiet ownership experience, or are these two
dimensions of safety largely independent of one another? The question is growing
more urgent rather than less. Vehicles are absorbing driver-assistance systems
that brake, steer, and warn on the driver's behalf, and electrification is
replacing familiar mechanical failures with unfamiliar electrical and thermal
ones. Both shifts change what can break and how dangerous the breakage is, and
neither is captured by a barrier impact. Understanding where the laboratory score
and the ownership record agree, where they diverge, and which kinds of vehicles
sit in the gap between them would help buyers read a window sticker more
accurately, and would help regulators decide where to look next.</p>

<h2>Ten questions this project sets out to answer</h2>
<ol class="questions">
  <li>Does a higher crash-test star rating correspond to fewer safety complaints
      once a vehicle is on the road?</li>
  <li>Which vehicle systems generate the most owner complaints, and which generate
      the most <em>dangerous</em> complaints?</li>
  <li>Are some vehicle segments — trucks, sport-utility vehicles, sedans, minivans —
      systematically more prone to fire, crash, or injury reports than others?</li>
  <li>How does estimated rollover probability relate to a vehicle's overall safety
      score, and does it predict real-world harm?</li>
  <li>Do electrified vehicles fail differently from petrol vehicles, and are their
      failures more or less harmful when they occur?</li>
  <li>How long after an incident do owners actually report it, and does that delay
      differ by how serious the incident was?</li>
  <li>At what point in a vehicle's life do safety problems tend to emerge?</li>
  <li>Do complaints reliably precede recalls, and how much warning does the
      complaint record give before formal action is taken?</li>
  <li>Can vehicles be grouped into meaningful safety profiles using their combined
      testing and ownership records, rather than by manufacturer or body style?</li>
  <li>Which measurable characteristics best predict whether a reported problem will
      involve a crash, a fire, or an injury?</li>
</ol>
"""

# ----------------------------------------------------------------------
# DATA PREP / EDA
# ----------------------------------------------------------------------

FIGURES = [
    ("fig01_complaints_per_year.png", "Complaints filed per year",
     "Complaint volume climbs steadily across the period before falling away in the "
     "most recent year. The final year is incomplete because the data were gathered "
     "part-way through it, so that decline reflects collection timing rather than "
     "safer vehicles."),
    ("fig02_complaints_by_segment.png", "Complaint volume by segment",
     "Sport-utility vehicles dominate the record with roughly half of all complaints, "
     "followed at a distance by sedans. Raw volume mostly tracks how many of each "
     "segment are on the road, so these counts are a measure of exposure rather than "
     "of risk."),
    ("fig03_harm_level.png", "Severity of reported incidents",
     "The overwhelming majority of complaints describe a fault with no crash and no "
     "fire, while fatal reports number in the dozens. A logarithmic axis is used "
     "because the categories differ by more than three orders of magnitude."),
    ("fig04_top_components.png", "Systems drawing the most complaints",
     "Electrical systems, powertrain, engine, and steering account for the largest "
     "share of reports. The prominence of electrical faults is notable given how "
     "little of a crash test addresses electronics."),
    ("fig05_crash_rate_by_component.png", "Which faults actually end in a crash",
     "Ranking systems by the share of their complaints that involved a collision "
     "reorders the picture entirely, lifting less common but more dangerous failures "
     "to the top. The highest-risk system is highlighted to separate it from the rest "
     "of the ranking."),
    ("fig06_reporting_lag.png", "Delay between incident and complaint",
     "Almost half of all complaints are filed within a week of the incident, but the "
     "distribution has a very long tail stretching to several years. The axis is "
     "logarithmic because that tail would otherwise compress the bulk of the data "
     "into a single bar."),
    ("fig07_vehicle_age.png", "Vehicle age when problems occur",
     "Reported problems peak in the first two years of a vehicle's life and decline "
     "steadily thereafter. Part of this pattern is real early-failure behaviour and "
     "part is that owners of newer vehicles are more likely to report at all."),
    ("fig08_fire_rate_by_segment.png", "Fire reports by segment",
     "Luxury vehicles and trucks show the highest proportion of fire-related "
     "complaints, while electrified vehicles sit at the lower end. This runs against "
     "the common assumption that battery vehicles are the most fire-prone segment on "
     "the road."),
    ("fig09_star_rating_distribution.png", "Distribution of crash-test ratings",
     "Most rated model-years in this collection hold four and a half to five stars, "
     "with very little spread below that. This compression is itself a finding: the "
     "rating scale separates modern vehicles from one another only weakly."),
    ("fig10_rollover_by_segment.png", "Rollover probability by segment",
     "Trucks and sport-utility vehicles carry roughly twice the estimated rollover "
     "probability of electric vehicles and sedans, reflecting their higher centres of "
     "gravity. The separation between segments is large and consistent."),
    ("fig11_crashrate_vs_rollover.png", "Crash involvement against rollover risk",
     "Each point is one model-year, sized by how many complaints it generated, and "
     "the cloud shows no clear upward trend. Vehicles most likely to tip are not the "
     "vehicles whose complaints most often involve a collision."),
    ("fig12_correlation_heatmap.png", "Correlations among safety measures",
     "The strongest relationships are between measures derived from the same source, "
     "such as crash rate and overall severity. Crash-test ratings correlate only "
     "weakly with any ownership-record measure, which is the central tension this "
     "project investigates."),
    ("fig13_top_model_years.png", "Model-years with the most complaints",
     "A small number of model-years account for a disproportionate share of the "
     "record. Identifying them early matters because they are where any predictive "
     "model will gain or lose most of its accuracy."),
    ("fig14_harm_mix_by_segment.png", "Outcome mix within each segment",
     "Normalising each segment to its own total removes the effect of fleet size and "
     "exposes differences in outcome. Hybrid and electric complaints are the most "
     "likely to involve a crash, while minivans are the least."),
    ("fig15_investigations_per_year.png", "Investigation records opened per year",
     "The bulk download extends back three decades and shows how regulatory activity "
     "has varied over time. This file provides historical depth that the current "
     "application programming interface does not expose."),
    ("fig16_recall_components.png", "Systems most often recalled",
     "Electrical systems, airbags, and reversing-camera equipment lead the recall "
     "record. Comparing this ranking against the complaint ranking shows where owner "
     "reports and formal regulatory action agree."),
]

DATAPREP = f"""<h2>Data Preparation and Exploratory Analysis</h2>

<p class="lede">Four separate collections of federal vehicle-safety data were
gathered, repaired, and joined into a single analysable table. This section
documents where every value came from, what was wrong with it, and what it looks
like once corrected.</p>

<h2>Where the data came from, and why</h2>

<p>The question driving this project needs two kinds of evidence that are not
normally stored together: what a laboratory says about a vehicle before it is
sold, and what its owners report after. Both are published by the U.S. National
Highway Traffic Safety Administration, the federal body responsible for vehicle
safety standards, and both are in the public domain. Data were gathered by two
independent routes so that the project does not depend on a single mechanism.</p>

<h3>Route 1 &mdash; Application Programming Interfaces</h3>

<p>An <b>API</b> (Application Programming Interface) is a published address that
returns structured data to a program rather than a page to a browser. Three
keyless NHTSA endpoints were queried in Python with the <code>requests</code>
library, sweeping 39 vehicle models across model years 2016 to 2024.</p>

<div class="table-wrap">
<table>
  <thead><tr><th>Source</th><th>Endpoint</th><th>What it provides</th></tr></thead>
  <tbody>
    <tr><td>Complaints</td>
        <td><code>api.nhtsa.gov/complaints/complaintsByVehicle</code></td>
        <td>Owner-filed defect reports: component, crash and fire flags, injuries,
            deaths, dates, narrative</td></tr>
    <tr><td>Recalls</td>
        <td><code>api.nhtsa.gov/recalls/recallsByVehicle</code></td>
        <td>Recall campaigns: affected component, consequence, remedy, park-outside
            warnings</td></tr>
    <tr><td>Safety Ratings</td>
        <td><code>api.nhtsa.gov/SafetyRatings/...</code></td>
        <td>Crash-test star ratings, rollover probability, driver-assistance
            equipment</td></tr>
  </tbody>
</table>
</div>

<p>A single core request looks like this, and returns JSON:</p>
<p><code>GET https://api.nhtsa.gov/complaints/complaintsByVehicle?make=honda&amp;model=accord&amp;modelYear=2020</code></p>

<p>Every response is cached to disk before parsing, so the gathering step is
resumable and reproducible without repeatedly loading the public service.</p>

<h3>Route 2 &mdash; Bulk file download</h3>

<p>The APIs expose current records only. To gain historical depth, the Office of
Defects Investigation flat file was downloaded directly and parsed from its
tab-delimited layout:</p>
<p><code>https://static.nhtsa.gov/odi/ffdd/inv/FLAT_INV.zip</code></p>

<div class="callout">
  <p><b>All code and all data are linked.</b></p>
  <ul>
    <li>Gathering: <a href="{BLOB}/code/fetch_data.py">fetch_data.py</a>
        &middot; Cleaning: <a href="{BLOB}/code/clean_data.py">clean_data.py</a>
        &middot; Visualisation: <a href="{BLOB}/code/eda_visualize.py">eda_visualize.py</a></li>
    <li>Raw data: <a href="{BLOB}/data/raw">data/raw/</a> &mdash;
        <a href="{RAW_GH}/data/raw/complaints_raw.csv.gz">complaints</a>,
        <a href="{RAW_GH}/data/raw/recalls_raw.csv.gz">recalls</a>,
        <a href="{RAW_GH}/data/raw/safety_ratings_raw.csv.gz">safety ratings</a>,
        <a href="{RAW_GH}/data/raw/investigations_raw.csv.gz">investigations</a></li>
    <li>Cleaned data: <a href="{BLOB}/data/clean">data/clean/</a> &mdash;
        <a href="{RAW_GH}/data/clean/complaints_clean.csv.gz">complaints</a>,
        <a href="{RAW_GH}/data/clean/vehicle_model_year.csv">joined model-year table</a></li>
    <li>Full cleaning log:
        <a href="{BLOB}/data/clean/cleaning_report.txt">cleaning_report.txt</a></li>
  </ul>
</div>

<h2>What was gathered</h2>

<div class="table-wrap">
<table>
  <thead><tr><th>Table</th><th>Source</th><th>Raw rows</th><th>Clean rows</th><th>Columns</th></tr></thead>
  <tbody>
    <tr><td>Complaints</td><td>API</td><td>84,135</td><td>83,967</td><td>21</td></tr>
    <tr><td>Recalls</td><td>API</td><td>1,662</td><td>1,662</td><td>15</td></tr>
    <tr><td>Safety ratings</td><td>API</td><td>505</td><td>421</td><td>27</td></tr>
    <tr><td>Investigations</td><td>Bulk download</td><td>154,380</td><td>138,271</td><td>10</td></tr>
    <tr><td><b>Joined model-year table</b></td><td>Derived</td><td>&mdash;</td><td>229</td><td>30</td></tr>
  </tbody>
</table>
</div>

<h2>What the data looked like before and after</h2>

{figure("snapshot_raw_complaints.png", "Raw complaints as returned by the API",
        "Dates arrive as text, the vehicle identification number is partial, and the "
        "component field packs several systems into one comma-separated string. "
        "Nothing here is typed, validated, or ready to analyse.")}

{figure("snapshot_raw_ratings.png", "Raw safety ratings, showing two disguised defects",
        "Missing star ratings are written as the text 'Not Rated' rather than left "
        "empty, and untested vehicles carry a rollover probability of exactly 0.0. "
        "A literal zero percent chance of rolling over is physically impossible, so "
        "that value is a placeholder masquerading as a measurement.")}

{figure("snapshot_clean_complaints.png", "Cleaned complaints",
        "Dates are parsed, the component field is reduced to a single primary system, "
        "and derived fields record severity, reporting delay, and vehicle age. The "
        "identification number is truncated to its manufacturer prefix for privacy.")}

{figure("snapshot_clean_model_year.png", "The joined model-year table",
        "Each row is one make, model, and model year, combining ownership-record "
        "measures with laboratory ratings. This is the table the later modules will "
        "cluster, decompose, and model.")}

<h2>Cleaning: what was wrong and what was done</h2>

<p>The cleaning and the exploration were carried out together rather than in
sequence, because most of the defects below were only visible once the data were
plotted. Every step is logged with before-and-after row counts in the
<a href="{BLOB}/data/clean/cleaning_report.txt">cleaning report</a>.</p>

<div class="table-wrap">
<table>
  <thead><tr><th>Problem found</th><th>Evidence</th><th>Action taken</th></tr></thead>
  <tbody>
    <tr><td>Missing values stored as text</td>
        <td><code>"Not Rated"</code> in every star-rating column; a blank space in
            the tip-test column</td>
        <td>Converted to true missing values, then handled explicitly</td></tr>
    <tr><td>Sentinel disguised as a measurement</td>
        <td>Rollover probability of exactly <code>0.0</code> on 46 vehicles &mdash;
            matching the unrated set exactly</td>
        <td>Recoded to missing; genuine values range from 6.6% to 20.8%</td></tr>
    <tr><td>Columns with no information</td>
        <td><code>RolloverRating2</code> and <code>RolloverPossibility2</code> hold one
            constant value in every row</td>
        <td>Dropped</td></tr>
    <tr><td>Impossible dates</td>
        <td>Incident dates as early as 1917 on vehicles built after 2015; two
            complaints filed before the incident occurred</td>
        <td>166 records removed as logically impossible</td></tr>
    <tr><td>Two different date formats</td>
        <td>Complaints use month/day/year; recalls use day/month/year</td>
        <td>Parsed separately with explicit formats rather than guessed</td></tr>
    <tr><td>Multi-label category field</td>
        <td>1,408 distinct comma-joined component combinations</td>
        <td>Reduced to the primary system, giving 41 usable categories</td></tr>
    <tr><td>Sentinel model year</td>
        <td>Bulk file uses <code>9999</code> for unknown</td>
        <td>16,108 such records removed</td></tr>
    <tr><td>Personal data</td>
        <td>Partial vehicle identification numbers present, 1,721 malformed</td>
        <td>Truncated to the three-character manufacturer prefix</td></tr>
    <tr><td>Join would duplicate outcomes</td>
        <td>Ratings are issued per body style, so a naive join copied one set of
            complaint counts onto several rows</td>
        <td>Ratings averaged to model-year level before joining</td></tr>
  </tbody>
</table>
</div>

<h3>Variables added</h3>
<ul>
  <li><b>Discretisation:</b> a five-level <code>harm_level</code> outcome
      (no crash or fire &rarr; fire &rarr; crash &rarr; injury &rarr; fatal), and a
      four-band rollover risk category.</li>
  <li><b>Derived measures:</b> reporting delay in days, vehicle age at incident,
      crash rate, fire rate, injuries per hundred complaints, and recall counts per
      model-year.</li>
  <li><b>Binary indicators:</b> whether forward-collision warning and lane-departure
      warning were standard equipment on every rated body style.</li>
</ul>

<div class="callout">
  <p><b>Result:</b> 99.8% of complaint records survived cleaning, and the final
  tables contain no missing values in any analysed column. Where records were
  removed, they were removed for a stated, checkable reason.</p>
</div>

<h2>Exploring the data</h2>

<p>Sixteen figures below examine the record from both directions: what owners
report, and what the laboratory measured.</p>

""" + "\n\n".join(figure(*f) for f in FIGURES) + f"""

<h2>What the exploration suggests</h2>

<p>Three patterns stand out before any modelling has been done. First, crash-test
ratings barely move: 85% of the rated model-years in this collection carry an
average of four and a half stars or better, so the measure that buyers rely on
separates modern vehicles only weakly. Second, star ratings show almost no
relationship with any ownership-record measure, while rollover probability is
moderately and negatively related to the overall score &mdash; the laboratory
number appears to describe a different dimension of safety from the one the
complaint record captures. Third, the segments that look safest on volume look
different once exposure is removed: hybrid and electric complaints are the most
likely to involve a crash, yet the least likely to involve a fire, which inverts
a common assumption.</p>

<p>These are associations in observational data, not causal findings, and the
complaint record is self-selected: it reflects who chose to report, not everything
that went wrong. The modelling modules that follow will test whether these
patterns survive more careful treatment.</p>
"""

CONCLUSIONS = """<h2>Conclusions</h2>
<p class="lede">This section closes the project's story in plain language, without
technical vocabulary, once all the analysis is complete.</p>
<div class="callout">
  <p><b>This section is completed in Module 5.</b> It will present what the work
  found, written for a general reader with no background in the subject, and
  supported by images rather than tables of numbers.</p>
</div>
<div class="placeholder">
  <p>The conclusions will answer the ten questions posed in the Introduction,
  state plainly which of them the data could and could not settle, and set out what
  a car buyer and a regulator should each take away from the result.</p>
</div>"""

ABOUT = f"""<h2>About Me</h2>

<p class="lede">I am a graduate student in <b>Data Science</b> with a background in
<b>Electronics and Communication Engineering</b> and professional experience across
software quality engineering and data engineering.</p>

<p>From <b>2022 to 2025 I worked as a Software Quality Engineer at Pegasystems</b>,
where I worked on enterprise software and developed experience with testing,
automation, APIs, data, and large-scale business systems.</p>

<p>During the summer of <b>2026 I worked as a Data Engineer intern at Denver
International Airport</b>, where I worked with data and technology projects in an
operational environment. This experience gave me an opportunity to apply my data
and engineering skills to real-world airport systems.</p>

<p>My current machine learning project brings together my technical background with
my long-standing interest in automobiles and vehicle safety. I am investigating a
simple but interesting question:</p>

<blockquote>
  <p>Do the crash-test scores used to sell vehicles tell us anything useful about
  the problems their owners actually experience?</p>
</blockquote>

<p>Rather than treating this as simply a machine learning modelling exercise, I am
working through the <b>full data science lifecycle</b> &mdash; from defining the
question and preparing the data to exploratory analysis, feature engineering,
modelling, evaluation, and communicating the results.</p>

<p>My undergraduate background in electronics and communications initially sparked
my interest in technology and real-world systems. My professional experience in
software engineering and data engineering, combined with my graduate studies in
Data Science, has given me a broader toolkit for investigating those systems
through data.</p>

<p>Outside of academics and work, I enjoy <b>automobiles, motorcycle riding, racing
games, and listening to jazz</b>.</p>

<h3>Elsewhere</h3>
<ul>
  <li><a href="https://www.linkedin.com/in/rajeshwar-rengeswaran/">Connect with me on LinkedIn</a></li>
  <li>Project code and data: <a href="{REPO}">{REPO}</a></li>
</ul>"""


PAGES = {
    "index.html": ("Introduction",
                   "Why laboratory crash-test scores may not describe how vehicles "
                   "actually fail in service.", INTRO),
    "dataprep_eda.html": ("DataPrep_EDA",
                          "Gathering, cleaning and exploring four federal vehicle "
                          "safety datasets.", DATAPREP),
    "clustering.html": ("Clustering", "Partitional and hierarchical clustering.",
        stub("Clustering", 2,
             "Grouping vehicles by their combined safety profile, rather than by "
             "manufacturer or body style.",
             "Partitional and hierarchical clustering will be applied to the joined "
             "model-year table to test whether vehicles fall into natural safety "
             "groupings, and whether those groupings correspond to anything a buyer "
             "would recognise.")),
    "pca.html": ("PCA", "Principal component analysis.",
        stub("PCA", 2,
             "Reducing many correlated safety measures to the few dimensions that "
             "carry most of the variation.",
             "Principal component analysis will be used to find which combinations "
             "of complaint, recall and crash-test measures explain the most variance, "
             "and whether laboratory and ownership measures load onto separate "
             "components.")),
    "naivebayes.html": ("NaiveBayes", "Naive Bayes classification.",
        stub("Naive Bayes", 3,
             "Predicting whether a reported problem turned into a crash, a fire, or "
             "an injury.",
             "Naive Bayes will be trained on the cleaned complaint records, using "
             "both the categorical component fields and the narrative text, with "
             "confusion matrices reported against a held-out test set.")),
    "dectrees.html": ("DecTrees", "Decision tree classification.",
        stub("Decision Trees", 3,
             "Finding the decision rules that best separate harmful complaints from "
             "routine ones.",
             "Decision trees will be fitted to the same prediction task as Naive "
             "Bayes, with tree depth tuned and the resulting rules visualised, since "
             "an interpretable rule set is valuable in a safety context.")),
    "svms.html": ("SVMs", "Support vector machines.",
        stub("Support Vector Machines", 4,
             "Testing whether a margin-based classifier separates safety outcomes "
             "more cleanly.",
             "Support vector machines will be applied with several kernels, linear "
             "and non-linear, and compared against one another and against the "
             "earlier classifiers using consistent accuracy measures.")),
    "regression.html": ("Regression", "Linear regression.",
        stub("Regression", 5,
             "Predicting continuous safety measures, such as complaint rate and "
             "rollover probability.",
             "Linear regression will model continuous outcomes from vehicle "
             "characteristics and equipment, with residuals examined to check whether "
             "the relationships found during exploration hold under a fitted model.")),
    "nn.html": ("NN", "Neural networks.",
        stub("Neural Networks", 5,
             "Testing whether a flexible model finds structure the simpler methods "
             "miss.",
             "A neural network will be trained on the same prediction tasks used "
             "earlier, and its accuracy compared against the simpler models to judge "
             "whether the additional complexity is justified.")),
    "conclusions.html": ("Conclusions", "What the project found.", CONCLUSIONS),
    "about.html": ("About Me", "About the author of this project.", ABOUT),
}


if __name__ == "__main__":
    for href, (title, desc, body) in PAGES.items():
        html = TEMPLATE.format(title=title, desc=desc, nav=nav_html(href),
                               body=body.strip(), repo=REPO)
        (ROOT / href).write_text(html, encoding="utf-8")
        print(f"  wrote {href:22s} {len(html)/1024:6.1f} KB")
    print(f"\n{len(PAGES)} pages written to {ROOT}")
