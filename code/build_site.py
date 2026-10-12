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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
<header class="masthead">
  <div class="wrap">
    <p class="course">Rajeshwar Rengeswaran, CSCI 5612 Machine Learning</p>
    <h1>Rated Safe, Reported Broken</h1>
    <p class="byline">What crash-test scores miss about how vehicles actually fail</p>
  </div>
</header>
<nav class="tabs" aria-label="Project sections">
  <div class="wrap">
{nav}
  </div>
</nav>
<main>
{body}
</main>
<footer><div class="wrap">
  <p>Rajeshwar Rengeswaran, CSCI 5612 Machine Learning project.</p>
  <p>Data published by the <a href="https://www.nhtsa.gov/">U.S. National Highway
  Traffic Safety Administration</a>. All code and data for this project are public at
  <a href="{repo}">github.com/Rajeshwar987/csci5612-project</a>.</p>
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
and still accumulate a long record of dangerous faults in ordinary service. The
rating is not dishonest about this, because it never claimed to cover it. The
difficulty is that the score is the only safety number most buyers ever see, so
it ends up carrying meaning it was never built to hold.</p>

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
and driven by whatever owners happened to notice and bother to report. That
dependence on human reporting is both its weakness and its reach. No laboratory
could afford to drive a million vehicles for a decade, but a million owners do
it anyway, and some fraction of them write down what went wrong. The resulting
archive is messy, uneven and irreplaceable.</p>

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
accurately, and would help regulators decide where to look next. Neither record
is wrong on its own terms. Each simply answers a question the other never
asked, and nobody has been required to read them together. Doing so is the
whole purpose of the work that follows.</p>

<p>There is also a timing problem buried in how this information reaches people.
A crash-test score is fixed at launch, published before a single example has
been sold, and it never changes afterwards. The ownership record moves in the
opposite direction: it is empty on the day the rating appears and accumulates
for as long as the vehicle stays on the road. The two therefore cross over at
some point in a vehicle's life, after which the score describes a machine that
no longer resembles the one being bought. This matters most for used vehicles,
which account for the large majority of cars changing hands, and which are
bought by people with the least money to absorb a serious fault. A buyer
comparing two eight-year-old vehicles is handed a number earned when both were
new, while the record of what actually went wrong with them sits in a federal
database that almost nobody consults. The gap is not that the rating is wrong.
It is that the rating is old, and the thing that has aged alongside the vehicle
is never shown next to it. Dealers are under no obligation to raise the
subject, and few buyers know the record exists at all. A rating that cannot
change and a record that only grows will drift apart by design, not by
accident. Closing that drift does not require a new test, only a willingness to
read the second record alongside the first.</p>

<figure>
  <img src="assets/img/crashtest_pole.jpg"
       alt="A vehicle positioned against a rigid pole for a side-impact crash test">
  <figcaption><b>A side-pole impact test.</b> A vehicle is pulled sideways into a
  narrow rigid pole, concentrating the entire force of the collision into a
  single point beside the driver. Photograph by the U.S. National Highway Traffic
  Safety Administration, a public-domain federal work.</figcaption>
</figure>

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

CLUSTERING = f"""<h2>Clustering</h2>

<p class="lede">Crash-test ratings sort vehicles into five stars. This section
ignores that sorting entirely and asks a different question: if the only thing
known about a vehicle were its measured safety record, what groups would the
data put it in on its own?</p>

<h2>Overview</h2>

<p>Clustering is unsupervised learning. Nothing is labelled, nothing is being
predicted, and the algorithm is never told what a good answer looks like. It
receives a set of points described by numbers and is asked to find points that
sit close together. What "close" means has to be defined before anything can
happen, which is why a distance metric is the first decision rather than a
detail, and why the same data can produce different groupings under different
definitions of distance. The value of the exercise is that any structure found
is a property of the measurements themselves, not an artefact of categories
that a human imposed beforehand.</p>

<p>Two families of algorithm are used here. <b>Partitional</b> clustering, of
which k-means is the standard example, demands the number of groups in advance.
It places k centres, assigns each vehicle to the nearest one, moves each centre
to the mean of the vehicles that chose it, and repeats until nothing moves.
It is fast and it always returns exactly k groups, but it has to be told k,
it assumes groups are roughly round and comparable in size, and it can settle
into a poor arrangement depending on where the centres started.
<b>Hierarchical</b> clustering makes no such demand. Every vehicle begins as
its own cluster and the two closest clusters are merged repeatedly until one
remains, recording the entire sequence as a tree. The number of groups is
chosen afterwards by cutting that tree at a chosen height, which means a single
run shows the structure at every possible resolution at once.</p>

{figure("concept_clustering_types.png",
        "Two ways of forming groups",
        "k-means commits to a fixed number of groups and assigns every point to "
        "exactly one of them, shown here with the cluster centres marked. "
        "Hierarchical clustering instead records every merge as a tree, leaving "
        "the number of groups to be chosen afterwards by cutting it.")}

<p>The distance metric matters more than it first appears. <b>Euclidean
distance</b> is ordinary straight-line distance and is what k-means minimises;
it responds to magnitude, so a vehicle with large values on many measures sits
far from one with small values. <b>Cosine distance</b> is one minus the cosine
of the angle between two vehicles treated as vectors, and it measures whether
two records have the same <i>shape</i> regardless of magnitude. The assignment
specifies cosine for the hierarchical clustering, and that choice fits the
question: it groups vehicles whose safety measures rise and fall together, so
a heavily complained-about model and a lightly complained-about one can land
together if their problems are proportionally alike. Running the two methods
under two different definitions of distance is also a check. Groups that
survive both are structural; groups that appear under only one are an artefact
of the metric.</p>

{figure("concept_distance_metrics.png",
        "What the two distance metrics measure",
        "Euclidean distance is the length of the line between two points and "
        "therefore grows with magnitude. Cosine distance measures only the angle "
        "between the directions, so two vehicles with proportionally similar "
        "safety profiles are close even when one has far more complaints.")}

<p>The plan for discovery is to cluster on measured outcomes alone, then look
at what the resulting groups turn out to contain. Body segment, manufacturer,
model year and star rating are all removed before fitting and held back. If
clusters built purely from complaint rates, recall counts and rollover figures
turn out to align with segment or rating, that alignment was discovered rather
than assumed. If they cut across those categories instead, the data is saying
that the usual way of describing a vehicle is not the way its safety record
behaves.</p>

<h2>Data Prep</h2>

<p>Clustering accepts <b>only unlabeled numeric data</b>. Every value must be a
number, because distances cannot be computed on text, and no column may carry
the answer, because an unsupervised method that can see a label is no longer
unsupervised. The joined model-year table from Module 1 therefore cannot be
used as it stands: it contains make, model, segment, the top complaint
component, and two categorical bands, all of which are discarded here and held
aside purely to interpret the result afterwards.</p>

<p>Three further preparation steps were necessary:</p>
<ul>
  <li><b>Raw counts were dropped.</b> Crash, fire, injury and death counts all
      scale with complaint volume. Keeping them would have let fleet size
      dominate every distance in the calculation, so the rates derived from them
      were kept instead.</li>
  <li><b>Skewed counts were log-transformed.</b> Complaint volume, recalls and
      investigations span three orders of magnitude. Without a log transform the
      handful of very high-volume vehicles would set the geometry for everyone
      else.</li>
  <li><b>Every feature was standardised</b> to mean 0 and standard deviation 1.
      This is not optional. The features arrive on wildly different scales, and
      an unscaled star rating running 1 to 5 would be overwhelmed by a
      complaint count running into the hundreds purely because of its units.</li>
</ul>

<p>The result is a matrix of <b>229 vehicles by 16 numeric features</b>, with no
labels of any kind. The three images below show the same data before the
transformation, part-way through it, and after it.</p>

{figure("sample_before_transform.png",
        "BEFORE: the Module 1 model-year table",
        "Text columns sit beside numeric ones, and the numeric columns are on "
        "completely different scales: a crash rate between 0 and 1 next to a "
        "complaint count in the hundreds next to a star rating from 1 to 5. "
        "Nothing here could be handed to a distance calculation.")}

{figure("sample_clustering_logged.png",
        "AFTER the first step: labels removed, skewed counts logged",
        "Every text column is gone and the three heavily skewed count columns "
        "have been log-transformed, which is why they now read as values near 5 "
        "or 6 rather than in the hundreds. The scales are still mismatched, so "
        "the data is not yet usable.")}

{figure("sample_clustering_input.png",
        "AFTER the second step: standardised, and ready",
        "Each column now has mean 0 and standard deviation 1, so every value "
        "reads as a number of standard deviations from average and positive and "
        "negative figures appear throughout. This is the complete input: the "
        "algorithms see nothing else.")}

<div class="callout">
  <p><b>Data and code.</b>
  Clustering input matrix:
  <a href="{RAW_GH}/data/clean/clustering_sample.csv">clustering_sample.csv</a> &middot;
  Resulting labels joined back to each vehicle:
  <a href="{RAW_GH}/data/clean/cluster_assignments.csv">cluster_assignments.csv</a> &middot;
  Silhouette sweep:
  <a href="{RAW_GH}/data/clean/kmeans_silhouette_sweep.csv">kmeans_silhouette_sweep.csv</a></p>
  <p>Code: <a href="{BLOB}/code/clustering_analysis.py">clustering_analysis.py</a>
  (full script) &middot;
  <a href="{BLOB}/notebooks/clustering.ipynb">clustering.ipynb</a>
  (annotated notebook with outputs) &mdash; Python, using scikit-learn for
  k-means and silhouette scoring and SciPy for the cosine-distance hierarchical
  linkage and dendrogram.</p>
</div>

<h2>Code</h2>

<p>The analysis is written in Python in a single script,
<a href="{BLOB}/code/clustering_analysis.py">clustering_analysis.py</a>. k-means
comes from <code>sklearn.cluster.KMeans</code>, run with 25 random restarts at
each k so the reported solution is not an unlucky initialisation. Hierarchical
clustering uses <code>scipy.cluster.hierarchy.linkage</code> on a condensed
<code>pdist</code> matrix computed with <code>metric="cosine"</code>. Average
linkage is used because Ward's method, the more common default, is defined only
for squared Euclidean distance and cannot be combined with cosine at all.
Silhouette scores come from <code>sklearn.metrics</code>, computed on the
precomputed cosine distance matrix for the hierarchical solutions so that each
method is scored under its own metric rather than a borrowed one.</p>

<h2>Results</h2>

<h3>Choosing k</h3>

<p>k-means was run for every k from 2 to 15 and scored by mean silhouette width,
which measures how much closer each vehicle sits to its own cluster than to the
next nearest one. The naive approach is to take the highest score, and here that
would have been actively misleading: the curve keeps drifting upward as k grows,
and the nominal maximum fell at k = 15, the edge of the tested range, where the
smallest cluster contained three vehicles. That rising pattern is the signature
of a dataset with no clean separation. The algorithm is slicing a continuum into
ever smaller pieces, and smaller pieces always look tidier.</p>

<p>Two corrections were applied. For k-means, only interior local maxima of the
curve were treated as candidates, which gave k = 7, 10 and 13. For the
hierarchical solutions, any k producing a singleton cluster was discarded before
reading off the peak, because a cluster of one vehicle has a silhouette width of
1 by definition and rewards the algorithm for stranding outliers. After those
corrections both methods point to the same answer independently: <b>k = 7</b> is
a local maximum for k-means at 0.225 and the global maximum for cosine-based
hierarchical clustering at 0.304.</p>

{figure("fig17_silhouette_by_k.png",
        "Silhouette score against k for both methods",
        "Both curves are low in absolute terms, peaking near 0.3 where values "
        "above 0.5 would indicate clean separation. The hierarchical curve under "
        "cosine distance sits consistently above the Euclidean k-means curve, "
        "and both identify k = 7.")}

{figure("fig18_silhouette_widths.png",
        "Per-vehicle silhouette widths at k = 7",
        "Each horizontal bar is one vehicle, grouped by cluster and sorted "
        "within it. Several clusters contain vehicles with negative widths, "
        "meaning those vehicles sit closer to a neighbouring cluster than to "
        "their own, which is the visual signature of overlapping groups.")}

<h3>Comparing three values of k</h3>

{figure("fig19_kmeans_k_comparison.png",
        "k-means at k = 3, k = 5 and k = 7, drawn in the plane of the first two principal components",
        "At k = 3 the algorithm splits the cloud into three broad slabs and "
        "scores worst, at 0.176. Moving to k = 7 isolates the small, genuinely "
        "distinct groups at the edges of the distribution, which is where the "
        "improvement in score comes from.")}

<h3>The hierarchical tree</h3>

{figure("fig20_dendrogram.png",
        "The full hierarchical tree under cosine distance",
        "Every one of the 229 vehicles enters at the bottom and merges upward "
        "until a single cluster remains. The red line marks the cut at k = 7, "
        "and the uneven branch widths show that the groups differ considerably "
        "in size.")}

{figure("fig20b_dendrogram_truncated.png",
        "The same tree truncated to its final 20 merges",
        "Collapsing the lower branches makes the top of the tree readable, with "
        "each leaf annotated by the number of vehicles beneath it. Reading the "
        "largest vertical gap excluding the root merge suggests a cut at k = 6, "
        "one below the silhouette answer.")}

<h3>Do the two methods agree?</h3>

{figure("fig21_kmeans_vs_hclust.png",
        "Agreement between the k-means and hierarchical solutions",
        "Each cell counts vehicles assigned to a given pair of clusters by the "
        "two methods. An adjusted Rand index of 0.638 indicates substantial but "
        "incomplete agreement, which is notable given that the methods use "
        "different distance metrics and entirely different procedures.")}

<h3>What the clusters contain</h3>

{figure("fig22_cluster_profiles.png",
        "What distinguishes each cluster",
        "Each cell is a cluster's mean on one standardised feature, so values "
        "read as standard deviations from the fleet average. Reading across a "
        "row gives that cluster's character, and the strongest colours mark the "
        "measures that set it apart.")}

<p>Labelling the seven groups by what actually separates them, with segment
composition checked only after fitting:</p>

<div class="table-wrap">
<table>
  <thead><tr><th>Cluster</th><th>n</th><th>What defines it</th><th>What it turned out to contain</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>10</td><td>Extreme regulatory attention: 18 recalls and 10 investigations on average, roughly ten times the fleet norm, with the lowest rollover probability and full driver-assistance fitment</td><td>7 of 10 are electric</td></tr>
    <tr><td>2</td><td>63</td><td>Oldest vehicles, highest complaint volume, lowest crash rate, and almost no driver assistance</td><td>Mixed: SUVs, sedans, minivans</td></tr>
    <tr><td>3</td><td>36</td><td>Highest rollover probability (0.18) and lowest crash-test ratings (4.1 stars)</td><td>Entirely trucks and large SUVs</td></tr>
    <tr><td>4</td><td>14</td><td>Fire rate of 6.5%, roughly five times every other cluster, and the slowest reporting at 23 days</td><td>Luxury-weighted, mixed</td></tr>
    <tr><td>5</td><td>75</td><td>Newest vehicles, universal collision and lane warning, near-perfect ratings, moderate everything else</td><td>The mainstream modern fleet</td></tr>
    <tr><td>6</td><td>26</td><td>Highest crash rate (19%) and highest injury rate (14.5 per 100 complaints) on low complaint volume</td><td>Sedan and hybrid weighted</td></tr>
    <tr><td>7</td><td>5</td><td>The only group rated 4 stars overall and 4 on rollover, with no driver assistance</td><td>Four sedans and one electric</td></tr>
  </tbody>
</table>
</div>

<h2>Conclusions</h2>

<p>Sorting vehicles by their safety records alone, without being told anything
about who made them or what they cost, produces a result worth knowing: there
is no such thing as a category of dangerous cars. Vehicles do not fall into a
safe group and an unsafe group. They shade into one another continuously, with
only a handful at the far edges standing clearly apart. Anyone hoping for a
simple list of models to avoid is looking for something the evidence does not
support, and that is probably the single most useful thing a car buyer could
take from this exercise.</p>

<p>What does separate vehicles is the shape of the machine and the equipment
bolted to it. The most clearly defined group that emerged contains trucks and
large sport-utility vehicles and nothing else. Nobody told the method that
these were trucks. It found them by noticing that tall, heavy vehicles tip over
more readily and score worse in crash testing, and that those two facts travel
together reliably enough to mark the group out. A second clear group is defined
entirely by having automatic braking and lane warning fitted as standard, which
in practice means it is made up of recent vehicles. Height and equipment turn
out to describe a vehicle's safety character far better than its badge does.</p>

<p>The most counterintuitive finding concerns complaints themselves. The
vehicles people complain about most often are not the vehicles whose complaints
describe the worst outcomes. One group has by far the highest volume of
complaints and among the lowest rates of actual harm, while another generates
relatively few complaints but a strikingly high share of them involve a crash
or an injury. Volume and seriousness pull in opposite directions. A shopper
who judges a model by how many complaints it has attracted is therefore
reading the wrong number, and so is anyone who assumes a quiet record means a
safe one.</p>

<p>One group deserves a note of caution rather than alarm. The vehicles
attracting the most regulatory attention, by a wide margin, are predominantly
electric, with roughly ten times the recalls and investigations of the typical
vehicle. This is easy to misread. These vehicles also hold the highest crash
ratings and the lowest rollover risk in the entire set, and their technology
allows many faults to be repaired remotely rather than at a workshop. Intense
scrutiny of a new technology is not the same thing as danger, and separating
those two ideas is exactly what a record like this is for.</p>
"""

PCA_TAB = f"""<h2>Principal Component Analysis</h2>

<p class="lede">Sixteen separate safety measures describe each vehicle. The
question here is whether they are really sixteen different things, or a handful
of underlying patterns wearing sixteen names.</p>

<h2>Overview</h2>

<p>Principal Component Analysis takes a set of correlated measurements and
rebuilds them as a new set of axes that are uncorrelated with one another and
ordered by how much of the data's variation each one carries. The first
principal component is the single direction through the data along which the
points are most spread out. The second is the direction of greatest remaining
spread that is perpendicular to the first, and so on. Each component is a
weighted blend of all the original variables, with the weights, called
loadings, saying how much each original measure contributes. Nothing is thrown
away in the rotation itself. The reduction happens afterwards, when the later
components, which carry little variation, are discarded.</p>

<p><b>Eigenvalues and eigenvectors</b> sit underneath all of this. For the
correlation matrix of the standardised data, an eigenvector is a direction that
the matrix stretches without rotating, and its eigenvalue is how much stretching
occurs. Translated into data terms, the eigenvectors are the candidate axes and
the eigenvalues are the variance captured along each. Sorting the eigenvectors
by their eigenvalues produces the principal components in order of importance,
which is the whole mechanism. Because the input was standardised, each original
variable contributes exactly 1 unit of variance, so an eigenvalue above 1 marks
a component carrying more information than any single original measure did on
its own.</p>

{figure("concept_pca_projection.png",
        "What the eigenvectors are, and what projection does",
        "The first eigenvector points along the direction in which the cloud is "
        "most stretched and the second lies perpendicular to it. Projecting every "
        "point onto the first axis alone collapses two dimensions into one while "
        "preserving most of the original spread.")}

<p><b>Dimensionality reduction</b> matters for three practical reasons. Visual
inspection is limited to two or three dimensions, so a sixteen-dimensional
dataset cannot be looked at directly without some projection. Many algorithms
degrade as dimensions grow, because in high-dimensional space the distance to
the nearest point and the distance to the furthest point converge, so "nearest"
stops carrying much meaning, and distance-based methods such as the clustering
in the previous section rest entirely on that meaning. And the data required to
cover a space grows exponentially with its dimension, so a sample that
comfortably describes two variables is sparse across sixteen. Correlated
variables also make models unstable, since the model cannot tell which of two
measures that move together is responsible for an effect.</p>

{figure("concept_dimensionality.png",
        "Why high-dimensional data is difficult",
        "As dimensions increase, the ratio between the closest and furthest pair "
        "of points climbs toward 1, so every point becomes roughly equidistant "
        "from every other. The sample size needed to maintain the same coverage "
        "grows exponentially alongside it.")}

<p><b>How PCA is used in this project.</b> It is put to three jobs here. First,
as a test of whether the sixteen safety measures are really sixteen separate
things: if a few components captured most of the variance, that would mean a
small number of hidden factors drive vehicle safety, and a single summary score
could in principle represent them. Second, as a way of seeing where the
laboratory measurements sit relative to the ownership record. Because every
component is a blend of all sixteen variables, the loadings show directly
whether star ratings and realised harm move together, move in opposition, or
are unrelated, which is the central question this project asks. Third, as a
drawing surface: the clustering in the previous section happened in sixteen
dimensions and cannot be looked at, so projecting those clusters onto the first
two components is what makes them visible, and whether they stay separated
under that projection is evidence about whether they were real.</p>

<h2>Data Prep</h2>

<p>PCA needs the same input shape as clustering: a <b>purely numeric matrix with
no labels</b>. The same 16 features and the same 229 vehicles are used, prepared
identically, so that the two sections are directly comparable. Standardisation
is even more important here than it was for clustering. PCA finds directions of
maximum variance, and variance depends on units, so an unstandardised variable
with a wide numeric range would capture the first component purely by being
measured on a larger scale. Standardising every column to mean 0 and standard
deviation 1 means the analysis runs on the correlation matrix rather than the
covariance matrix, and every variable gets an equal vote.</p>

{figure("sample_before_transform.png",
        "BEFORE: the Module 1 model-year table",
        "The starting point is the same table used everywhere else in this "
        "project, carrying text labels alongside numeric columns measured on "
        "entirely different scales. PCA cannot be run on this directly.")}

{figure("sample_pca_before.png",
        "BEFORE standardising: the numeric features in their own units",
        "Labels have been dropped and skewed counts logged, but the columns "
        "still have wildly unequal variances. Running PCA at this stage would "
        "hand the first component to whichever variable happens to be measured "
        "on the largest scale.")}

{figure("sample_pca_input.png",
        "AFTER standardising: the matrix PCA actually receives",
        "Every value is now expressed in standard deviations from that column's "
        "mean, so each variable contributes exactly one unit of variance and no "
        "column can dominate through its units. This is what makes the analysis "
        "run on the correlation matrix rather than the covariance matrix.")}

<div class="callout">
  <p><b>Data and code.</b>
  PCA input: <a href="{RAW_GH}/data/clean/pca_sample.csv">pca_sample.csv</a> &middot;
  Eigenvalues: <a href="{RAW_GH}/data/clean/pca_eigenvalues.csv">pca_eigenvalues.csv</a> &middot;
  Loadings: <a href="{RAW_GH}/data/clean/pca_components.csv">pca_components.csv</a> &middot;
  Vehicle scores: <a href="{RAW_GH}/data/clean/pca_scores.csv">pca_scores.csv</a></p>
  <p>Code: <a href="{BLOB}/code/pca_analysis.py">pca_analysis.py</a>
  (full script) &middot;
  <a href="{BLOB}/notebooks/pca.ipynb">pca.ipynb</a>
  (annotated notebook with outputs) &mdash; Python, using
  <code>sklearn.decomposition.PCA</code> with <code>StandardScaler</code>.</p>
</div>

<h2>Code</h2>

<p>The analysis is in <a href="{BLOB}/code/pca_analysis.py">pca_analysis.py</a>.
<code>StandardScaler</code> centres and scales the matrix, then
<code>sklearn.decomposition.PCA</code> is fitted with all components retained so
the full eigenvalue spectrum is available for the scree plot. Eigenvalues are
read from <code>explained_variance_</code> and the eigenvectors from
<code>components_</code>. The correlation circle multiplies each loading by the
square root of its eigenvalue, which converts loadings into correlations between
the original variable and the component and is what bounds every arrow inside
the unit circle.</p>

<h2>Results</h2>

<h3>How much structure is there?</h3>

{figure("fig23_scree.png",
        "Variance carried by each principal component",
        "The first component accounts for 24.2% of the total variance and the "
        "second for 14.8%. The curve declines gently rather than dropping off a "
        "cliff, which means there is no small set of components that captures "
        "almost everything.")}

{figure("fig24_eigenvalues.png",
        "Eigenvalues of the correlation matrix",
        "Six components have eigenvalues above 1, meaning each carries more "
        "variance than any single original variable contributed. Under the Kaiser "
        "criterion those six are worth retaining and the remaining ten are not.")}

<div class="table-wrap">
<table>
  <thead><tr><th>Measure</th><th>Result</th><th>Reading</th></tr></thead>
  <tbody>
    <tr><td>Variance in PC1</td><td>24.2%</td><td>No single dominant pattern</td></tr>
    <tr><td>Variance in PC1 + PC2</td><td>39.0%</td><td>A two-dimensional plot shows well under half the picture</td></tr>
    <tr><td>Eigenvalues above 1</td><td>6</td><td>Six components beat a single original variable</td></tr>
    <tr><td>Components for 90% variance</td><td>9 of 16</td><td>Only a 44% reduction in dimensions</td></tr>
    <tr><td>Components for 95% variance</td><td>12 of 16</td><td>Compression is poor</td></tr>
  </tbody>
</table>
</div>

<h3>What the components mean</h3>

{figure("fig25_loadings.png",
        "How each original variable builds each component",
        "Reading down a column shows what that component is made of, with "
        "positive and negative loadings pulling in opposite directions. PC1 sets "
        "complaint volume and vehicle age against crash rate and driver-assistance "
        "fitment; PC2 sets crash-test ratings against realised harm.")}

{figure("fig26_correlation_circle.png",
        "How each variable relates to the first two components",
        "Arrows pointing the same way are correlated, opposite directions are "
        "inversely related, and right angles mean unrelated. The severity cluster "
        "at upper right and the star-rating cluster at lower right sit nearly "
        "perpendicular to the complaint-volume arrow on the left.")}

<p>Reading the two leading components:</p>
<ul>
  <li><b>PC1 (24.2%) separates an older, high-volume, unassisted fleet from a
      newer, assisted one with a higher per-complaint harm rate.</b> Complaint
      volume (&minus;0.36), vehicle age (&minus;0.33) and recalls (&minus;0.23)
      load negatively, while crash rate (+0.37), severe share (+0.37) and the two
      driver-assistance indicators (+0.34, +0.33) load positively. It is
      essentially an age-and-exposure axis.</li>
  <li><b>PC2 (14.8%) sets laboratory ratings directly against real-world
      harm.</b> Overall stars (&minus;0.33), front-crash stars (&minus;0.32) and
      rollover stars (&minus;0.32) oppose severe share (+0.35), injuries per 100
      (+0.34) and crash rate (+0.30). This component is the central question of
      the whole project rendered as a single axis.</li>
</ul>

{figure("fig27_pca_projection_clusters.png",
        "The seven clusters projected onto the first two components",
        "The clusters were fitted on all 16 dimensions, so their separation here "
        "is evidence that two components retain genuine structure. The truck and "
        "large-SUV group and the high-severity group occupy clearly different "
        "regions, while the two largest clusters overlap substantially.")}

<h2>Conclusions</h2>

<p>The most useful result here is a negative one, and it goes to the heart of
what this project is asking. If vehicle safety were driven by a few underlying
qualities &mdash; some general notion of how well a car is built, say &mdash;
then the sixteen things measured here would mostly be different views of those
few qualities, and it would be possible to boil them down to two or three
numbers without losing much. That is not what happens. It takes nine of the
sixteen measures to preserve most of what makes one vehicle differ from
another. In plain terms, these really are sixteen separate things to know about
a car, and knowing some of them tells you remarkably little about the rest.
That is the clearest argument this project can offer for why a single safety
score, however carefully earned, leaves so much unsaid.</p>

<p>The second finding is sharper. When the measures are arranged by how they
vary together, crash-test performance and real-world harm land on opposite
sides of the same divide. They are not simply unrelated. Along the strongest
dimension on which both appear, they pull against each other: the vehicles
scoring best in the laboratory are not the ones whose owners report the fewest
serious problems, and in this collection they tend to be different vehicles
altogether. Module 1 noticed this as a weak correlation between star rating and
complaint measures. Seen this way it is not a weak signal at all, but a clear
separation between two kinds of safety that happen to share a name.</p>

<p>A third point concerns how much of this a picture can honestly show. Any
two-dimensional chart of this data, including the ones on this page, is
displaying well under half of what is actually there. That is worth stating
plainly rather than hiding, because it sets a limit on how far any visual
argument about vehicle safety can be pushed. The patterns visible in these
charts are real, but they are a summary of a far messier reality, and a
confident-looking plot is not the same thing as a complete one.</p>

<p>Taken together with the clustering, the message for a car buyer is
consistent. There is no hidden single number that captures how safe a vehicle
is, no shortcut that collapses the question into one score, and good
performance in one respect carries no promise of good performance in another. A
vehicle can be excellent at protecting people during a collision and still be
among the more troublesome to own, and those two facts sit in different
records that nobody is currently required to read side by side.</p>
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
    "clustering.html": ("Clustering",
                        "k-means and cosine-distance hierarchical clustering of "
                        "229 vehicles on their measured safety records.", CLUSTERING),
    "pca.html": ("PCA",
                 "Principal component analysis of sixteen vehicle safety measures, "
                 "with eigenvalues, loadings and projections.", PCA_TAB),
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
