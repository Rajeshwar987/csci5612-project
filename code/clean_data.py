"""
clean_data.py  --  CSCI 5612 Project, Module 1 Part 1
Cleans and prepares the raw NHTSA data gathered by fetch_data.py.

Every transformation is logged to data/clean/cleaning_report.txt with
before/after row counts so the cleaning process is fully transparent.

Inputs   data/raw/*.csv.gz
Outputs  data/clean/complaints_clean.csv.gz
         data/clean/safety_ratings_clean.csv
         data/clean/recalls_clean.csv
         data/clean/investigations_clean.csv.gz
         data/clean/vehicle_model_year.csv     <- joined, model-ready table
         data/clean/cleaning_report.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# Fixed reference date keeps the cleaning reproducible on later re-runs.
TODAY = pd.Timestamp("2026-09-20")

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def step(name, before, after):
    log(f"    {name:<52s} {before:>7,} -> {after:>7,}  ({before - after:+,})")


# ----------------------------------------------------------------------
# 1. COMPLAINTS
# ----------------------------------------------------------------------

def clean_complaints():
    log("\n[1] COMPLAINTS")
    df = pd.read_csv(RAW / "complaints_raw.csv.gz", low_memory=False)
    log(f"    raw shape: {df.shape}")
    n0 = len(df)

    n = len(df)
    df = df.drop_duplicates(subset="odiNumber")
    step("drop duplicate complaint IDs", n, len(df))

    # Complaint dates are MM/DD/YYYY.
    df["date_incident"] = pd.to_datetime(df["dateOfIncident"], format="%m/%d/%Y",
                                         errors="coerce")
    df["date_filed"] = pd.to_datetime(df["dateComplaintFiled"], format="%m/%d/%Y",
                                      errors="coerce")
    n = len(df)
    df = df.dropna(subset=["date_incident", "date_filed"])
    step("drop unparseable dates", n, len(df))

    # A complaint cannot be filed before the incident happened, and neither
    # date can lie in the future relative to the data pull.
    n = len(df)
    df = df[(df["date_incident"] <= df["date_filed"]) &
            (df["date_filed"] <= TODAY)]
    step("drop filed-before-incident / future dates", n, len(df))

    # An incident cannot predate the vehicle. Model years are released up to
    # one calendar year early, so year-1 is permitted.
    n = len(df)
    df = df[df["date_incident"].dt.year >= df["query_year"] - 1]
    step("drop incidents predating the model year", n, len(df))

    # Reporting lag beyond ~15 years indicates a mistyped incident year
    # rather than a genuine delayed report.
    df["days_to_report"] = (df["date_filed"] - df["date_incident"]).dt.days
    n = len(df)
    df = df[df["days_to_report"].between(0, 5475)]
    step("drop implausible reporting lags (>15 yrs)", n, len(df))

    # 'UNKNOWN MANUFACTURER' is a sentinel string, not a real value.
    sentinel = (df["manufacturer"] == "UNKNOWN MANUFACTURER").sum()
    df.loc[df["manufacturer"] == "UNKNOWN MANUFACTURER", "manufacturer"] = np.nan
    df["manufacturer"] = df["manufacturer"].fillna(
        df["query_make"].str.title() + " (manufacturer not stated)")
    log(f"    {'recoded UNKNOWN MANUFACTURER sentinel':<52s} {sentinel:>7,} rows")

    # 'components' is a comma-joined multi-label field (1,409 combinations).
    # The first entry is the primary system implicated.
    before_u = df["components"].nunique()
    df["component_primary"] = (df["components"].fillna("UNKNOWN OR OTHER")
                               .str.split(",").str[0].str.strip().str.upper())
    log(f"    {'components collapsed to primary system':<52s} "
        f"{before_u:>7,} -> {df['component_primary'].nunique():>7,} categories")

    # VINs are partial (11 of 17 characters). Only the World Manufacturer
    # Identifier is retained; the rest is dropped for privacy and because
    # malformed lengths make the remainder unreliable.
    df["vin_valid"] = df["vin"].fillna("").str.len().eq(11)
    df["wmi"] = df["vin"].where(df["vin_valid"]).str[:3]
    df["wmi"] = df["wmi"].fillna("UNK")
    df = df.drop(columns=["vin"])
    log(f"    {'VIN reduced to WMI prefix (privacy)':<52s} "
        f"{(~df['vin_valid']).sum():>7,} malformed flagged")

    df["vehicle_age_at_incident"] = (df["date_incident"].dt.year
                                     - df["query_year"]).clip(lower=0)

    # Discretized severity outcome, ordered from least to most harmful.
    df["harm_level"] = np.select(
        [df["numberOfDeaths"] > 0,
         df["numberOfInjuries"] > 0,
         df["crash"].astype(bool),
         df["fire"].astype(bool)],
        ["Fatal", "Injury", "Crash, no injury", "Fire, no crash"],
        default="No crash or fire",
    )
    df["is_severe"] = df["harm_level"].isin(
        ["Fatal", "Injury", "Crash, no injury", "Fire, no crash"])

    df["report_year"] = df["date_filed"].dt.year
    df["summary"] = df["summary"].fillna("").str.slice(0, 400)

    keep = ["odiNumber", "manufacturer", "query_make", "query_model",
            "query_year", "segment", "crash", "fire", "numberOfInjuries",
            "numberOfDeaths", "harm_level", "is_severe", "component_primary",
            "date_incident", "date_filed", "days_to_report", "report_year",
            "vehicle_age_at_incident", "wmi", "vin_valid", "summary"]
    df = df[keep].rename(columns={"query_make": "make", "query_model": "model",
                                  "query_year": "model_year",
                                  "numberOfInjuries": "injuries",
                                  "numberOfDeaths": "deaths"})

    assert df.drop(columns=["summary"]).notna().all().all(), "NAs remain in complaints"
    log(f"    FINAL: {df.shape}  (kept {len(df)/n0:.1%} of raw rows)")
    return df


# ----------------------------------------------------------------------
# 2. SAFETY RATINGS
# ----------------------------------------------------------------------

RATING_COLS = [
    "OverallRating", "OverallFrontCrashRating", "FrontCrashDriversideRating",
    "FrontCrashPassengersideRating", "OverallSideCrashRating",
    "SideCrashDriversideRating", "SideCrashPassengersideRating",
    "combinedSideBarrierAndPoleRating-Front",
    "combinedSideBarrierAndPoleRating-Rear", "sideBarrierRating-Overall",
    "RolloverRating", "SidePoleCrashRating",
]


def clean_ratings():
    log("\n[2] SAFETY RATINGS")
    df = pd.read_csv(RAW / "safety_ratings_raw.csv.gz", low_memory=False)
    log(f"    raw shape: {df.shape}")

    n = len(df)
    df = df.drop_duplicates(subset="VehicleId")
    step("drop duplicate VehicleId records", n, len(df))

    # RolloverRating2 / RolloverPossibility2 carry a single constant value for
    # every row and therefore hold no information.
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)
    log(f"    {'dropped zero-variance columns':<52s} {constant}")

    # Crash-test media URLs are kept aside for the website, not for modelling.
    media = [c for c in df.columns if c.endswith(("Picture", "Video"))]
    images = df[["Make", "Model", "ModelYear"] +
                [c for c in media if c.endswith("Picture")]].copy()
    df = df.drop(columns=media)

    # 'Not Rated' and blank strings are missing values written as text.
    for c in [c for c in RATING_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c].replace({"Not Rated": np.nan, " ": np.nan}),
                              errors="coerce")
    df["dynamicTipResult"] = df["dynamicTipResult"].replace({" ": np.nan})

    # RolloverPossibility is stored as 0.0 for untested vehicles. A literal
    # zero percent rollover probability is physically impossible, and the 0.0
    # rows match the 'Not Rated' rollover set exactly, confirming a sentinel.
    sentinel = (df["RolloverPossibility"] == 0.0).sum()
    df.loc[df["RolloverPossibility"] == 0.0, "RolloverPossibility"] = np.nan
    log(f"    {'recoded RolloverPossibility 0.0 sentinel -> NA':<52s} {sentinel:>7,} rows")

    # The overall star rating is the analysis target, so unrated vehicles
    # cannot contribute and are removed rather than imputed.
    n = len(df)
    df = df.dropna(subset=["OverallRating"])
    step("drop vehicles with no overall rating", n, len(df))

    # Remaining gaps in secondary ratings are filled with that vehicle's
    # overall rating, the most defensible neighbour value.
    for c in [c for c in RATING_COLS if c in df.columns]:
        df[c] = df[c].fillna(df["OverallRating"])
    df["RolloverPossibility"] = df["RolloverPossibility"].fillna(
        df["RolloverPossibility"].median())
    df["dynamicTipResult"] = df["dynamicTipResult"].fillna("Not Tested")

    df["rollover_risk_band"] = pd.cut(
        df["RolloverPossibility"], bins=[0, 0.10, 0.15, 0.20, 1.0],
        labels=["Low (<10%)", "Moderate (10-15%)", "Elevated (15-20%)", "High (>20%)"])
    df["rollover_risk_band"] = df["rollover_risk_band"].astype(str)

    for c in ["NHTSAForwardCollisionWarning", "NHTSALaneDepartureWarning",
              "NHTSAElectronicStabilityControl"]:
        df[c] = df[c].fillna("No").replace({"Standard & Optional": "Standard"})

    df = df.rename(columns={"query_make": "make", "query_model": "model",
                            "query_year": "model_year"})
    df = df.drop(columns=["Make", "Model", "ModelYear"], errors="ignore")

    assert df.notna().all().all(), "NAs remain in ratings"
    log(f"    FINAL: {df.shape}")
    return df, images


# ----------------------------------------------------------------------
# 3. RECALLS
# ----------------------------------------------------------------------

def clean_recalls():
    log("\n[3] RECALLS")
    df = pd.read_csv(RAW / "recalls_raw.csv.gz", low_memory=False)
    log(f"    raw shape: {df.shape}")

    n = len(df)
    df = df.drop_duplicates(subset=["NHTSACampaignNumber", "query_make",
                                    "query_model", "query_year"])
    step("drop duplicate campaign/vehicle rows", n, len(df))

    # NHTSAActionNumber is 92% empty and Notes is identical boilerplate.
    df = df.drop(columns=["NHTSAActionNumber", "Notes"], errors="ignore")
    log(f"    {'dropped NHTSAActionNumber (92% empty) and Notes':<52s}")

    # Recall dates are DD/MM/YYYY, unlike the MM/DD/YYYY complaint dates.
    df["date_reported"] = pd.to_datetime(df["ReportReceivedDate"],
                                         format="%d/%m/%Y", errors="coerce")
    n = len(df)
    df = df.dropna(subset=["date_reported"])
    step("drop unparseable recall dates", n, len(df))

    for c in ["parkIt", "parkOutSide", "overTheAirUpdate"]:
        df[c] = df[c].fillna(False).astype(bool)
    log(f"    {'filled missing park/OTA flags with False':<52s}")

    df["component_primary"] = (df["Component"].fillna("UNKNOWN")
                               .str.split(":").str[0].str.split(",").str[0]
                               .str.strip().str.upper())

    for c in ["Summary", "Consequence", "Remedy"]:
        df[c] = df[c].fillna("").str.slice(0, 300)

    df = df.rename(columns={"query_make": "make", "query_model": "model",
                            "query_year": "model_year"})
    df = df.drop(columns=["Make", "Model", "ModelYear", "ReportReceivedDate"],
                 errors="ignore")

    assert df.notna().all().all(), "NAs remain in recalls"
    log(f"    FINAL: {df.shape}")
    return df


# ----------------------------------------------------------------------
# 4. INVESTIGATIONS (bulk download)
# ----------------------------------------------------------------------

def clean_investigations():
    log("\n[4] INVESTIGATIONS (bulk file)")
    df = pd.read_csv(RAW / "investigations_raw.csv.gz", low_memory=False)
    log(f"    raw shape: {df.shape}")

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    n = len(df)
    # 9999 is the flat file's 'unknown model year' sentinel.
    df = df[df["YEAR"].between(1990, 2026)]
    step("drop unknown/sentinel model years (incl. 9999)", n, len(df))

    df["date_opened"] = pd.to_datetime(df["ODATE"], format="%Y%m%d", errors="coerce")
    df["date_closed"] = pd.to_datetime(df["CDATE"], format="%Y%m%d", errors="coerce")
    n = len(df)
    df = df.dropna(subset=["date_opened"])
    step("drop records with no open date", n, len(df))

    df["is_open"] = df["date_closed"].isna()
    df["days_open"] = (df["date_closed"].fillna(TODAY) - df["date_opened"]).dt.days
    n = len(df)
    df = df[df["days_open"] >= 0]
    step("drop investigations closed before opening", n, len(df))

    df["MAKE"] = df["MAKE"].fillna("UNKNOWN").str.upper().str.strip()
    df["MODEL"] = df["MODEL"].fillna("UNKNOWN").str.upper().str.strip()
    df["COMPNAME"] = df["COMPNAME"].fillna("UNKNOWN").str.upper().str.strip()
    df["component_primary"] = df["COMPNAME"].str.split(":").str[0].str.strip()
    df["SUBJECT"] = df["SUBJECT"].fillna("").str.slice(0, 200)

    keep = ["NHTSA_ACTION_NUMBER", "MAKE", "MODEL", "YEAR", "component_primary",
            "date_opened", "date_closed", "is_open", "days_open", "SUBJECT"]
    df = df[keep].rename(columns={"NHTSA_ACTION_NUMBER": "action_number",
                                  "MAKE": "make", "MODEL": "model", "YEAR": "model_year",
                                  "SUBJECT": "subject"})
    df["model_year"] = df["model_year"].astype(int)
    df["date_closed"] = df["date_closed"].astype("string").fillna("still open")

    log(f"    FINAL: {df.shape}")
    return df


# ----------------------------------------------------------------------
# 5. JOINED MODEL-YEAR TABLE
# ----------------------------------------------------------------------

def build_model_year_table(complaints, ratings, recalls):
    """One row per make/model/model-year: the table later modules will model."""
    log("\n[5] JOINED MODEL-YEAR TABLE")

    agg = complaints.groupby(["make", "model", "model_year", "segment"]).agg(
        n_complaints=("odiNumber", "count"),
        n_crash=("crash", "sum"),
        n_fire=("fire", "sum"),
        total_injuries=("injuries", "sum"),
        total_deaths=("deaths", "sum"),
        median_days_to_report=("days_to_report", "median"),
        mean_vehicle_age=("vehicle_age_at_incident", "mean"),
        pct_severe=("is_severe", "mean"),
    ).reset_index()

    agg["crash_rate"] = agg["n_crash"] / agg["n_complaints"]
    agg["fire_rate"] = agg["n_fire"] / agg["n_complaints"]
    agg["injuries_per_100_complaints"] = 100 * agg["total_injuries"] / agg["n_complaints"]

    top_component = (complaints.groupby(["make", "model", "model_year"])["component_primary"]
                     .agg(lambda s: s.value_counts().idxmax())
                     .rename("top_component").reset_index())
    agg = agg.merge(top_component, on=["make", "model", "model_year"], how="left")

    rec = (recalls.groupby(["make", "model", "model_year"])
           .agg(n_recalls=("NHTSACampaignNumber", "nunique"),
                n_park_it_recalls=("parkIt", "sum"))
           .reset_index())
    agg = agg.merge(rec, on=["make", "model", "model_year"], how="left")
    agg[["n_recalls", "n_park_it_recalls"]] = \
        agg[["n_recalls", "n_park_it_recalls"]].fillna(0).astype(int)

    # NHTSA rates each body style / drivetrain separately, so a single
    # model-year can carry several rating records. Joining them directly would
    # copy one set of complaint counts onto every trim and silently duplicate
    # the outcome. Ratings are therefore collapsed to one row per model-year
    # first: stars and rollover probability are averaged, and a driver-assist
    # feature counts as present only if every rated trim offers it as standard.
    n_rating_rows = len(ratings)
    r = ratings.copy()
    for c in ["NHTSAElectronicStabilityControl", "NHTSAForwardCollisionWarning",
              "NHTSALaneDepartureWarning"]:
        r[c] = (r[c] == "Standard").astype(int)

    ratings_my = r.groupby(["make", "model", "model_year"]).agg(
        n_rated_trims=("VehicleId", "nunique"),
        OverallRating=("OverallRating", "mean"),
        OverallFrontCrashRating=("OverallFrontCrashRating", "mean"),
        OverallSideCrashRating=("OverallSideCrashRating", "mean"),
        RolloverRating=("RolloverRating", "mean"),
        RolloverPossibility=("RolloverPossibility", "mean"),
        InvestigationCount=("InvestigationCount", "max"),
        esc_standard=("NHTSAElectronicStabilityControl", "min"),
        has_fcw=("NHTSAForwardCollisionWarning", "min"),
        has_ldw=("NHTSALaneDepartureWarning", "min"),
    ).reset_index()
    step("collapse rating trims to one row per model-year",
         n_rating_rows, len(ratings_my))

    before = len(agg)
    agg = agg.merge(ratings_my, on=["make", "model", "model_year"], how="inner")
    step("inner-join complaints aggregate to crash-test ratings", before, len(agg))

    agg["rollover_risk_band"] = pd.cut(
        agg["RolloverPossibility"], bins=[0, 0.10, 0.15, 0.20, 1.0],
        labels=["Low (<10%)", "Moderate (10-15%)", "Elevated (15-20%)",
                "High (>20%)"]).astype(str)
    # Averaged stars are continuous; a rounded band gives later modules a
    # ready-made classification target.
    agg["rating_class"] = np.where(agg["OverallRating"] >= 4.5,
                                   "5-star", "4-star or below")

    agg = agg.round(4)

    assert agg.notna().all().all(), "NAs remain in joined table"
    log(f"    FINAL: {agg.shape}")
    return agg


if __name__ == "__main__":
    log("=" * 78)
    log("CLEANING REPORT - NHTSA vehicle safety data")
    log(f"reference date: {TODAY.date()}")
    log("=" * 78)

    complaints = clean_complaints()
    ratings, images = clean_ratings()
    recalls = clean_recalls()
    investigations = clean_investigations()
    model_year = build_model_year_table(complaints, ratings, recalls)

    complaints.to_csv(CLEAN / "complaints_clean.csv.gz", index=False)
    ratings.to_csv(CLEAN / "safety_ratings_clean.csv", index=False)
    recalls.to_csv(CLEAN / "recalls_clean.csv", index=False)
    investigations.to_csv(CLEAN / "investigations_clean.csv.gz", index=False)
    model_year.to_csv(CLEAN / "vehicle_model_year.csv", index=False)
    images.to_csv(CLEAN / "crash_test_images.csv", index=False)

    log("\n" + "=" * 78)
    log("WROTE:")
    for f in sorted(CLEAN.glob("*")):
        if f.suffix in (".csv", ".gz"):
            log(f"    {f.name:<34s} {f.stat().st_size/1024:>9,.0f} KB")
    (CLEAN / "cleaning_report.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nreport -> {CLEAN / 'cleaning_report.txt'}")
