"""
fetch_data.py  --  CSCI 5612 Project, Module 1 Part 1
Gathers raw vehicle-safety data for the project.

Two independent gathering strategies are used, as the assignment requires:

  (1) API GATHERING  -- three keyless public endpoints served by the
      U.S. National Highway Traffic Safety Administration (NHTSA):

        Complaints     https://api.nhtsa.gov/complaints/complaintsByVehicle
        Recalls        https://api.nhtsa.gov/recalls/recallsByVehicle
        Safety Ratings https://api.nhtsa.gov/SafetyRatings/...

      Example GET used by this script:
        https://api.nhtsa.gov/complaints/complaintsByVehicle?make=honda&model=accord&modelYear=2020

  (2) BULK DOWNLOAD -- the NHTSA Office of Defects Investigation flat file
        https://static.nhtsa.gov/odi/ffdd/inv/FLAT_INV.zip

Every raw response is cached to data/raw/cache/ so the run is resumable and
so the gathering is reproducible without re-hitting the APIs.

Outputs (raw, uncleaned):
    data/raw/complaints_raw.csv
    data/raw/recalls_raw.csv
    data/raw/safety_ratings_raw.csv
    data/raw/investigations_raw.csv
"""

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = RAW / "cache"
RAW.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "csci5612-student-project/1.0"}
PAUSE = 0.12          # polite delay between API calls
MAX_RETRIES = 3

YEARS = list(range(2016, 2025))

# A deliberately broad slice of the U.S. fleet: sedans, SUVs, trucks,
# minivans, luxury marques, and electrified models. Segment is recorded here
# so it can be used as a grouping variable during analysis.
VEHICLES = [
    ("honda", "accord", "Sedan"),       ("honda", "civic", "Sedan"),
    ("toyota", "camry", "Sedan"),       ("toyota", "corolla", "Sedan"),
    ("nissan", "altima", "Sedan"),      ("hyundai", "sonata", "Sedan"),
    ("chevrolet", "malibu", "Sedan"),   ("subaru", "legacy", "Sedan"),
    ("toyota", "rav4", "SUV"),          ("honda", "cr-v", "SUV"),
    ("ford", "explorer", "SUV"),        ("chevrolet", "equinox", "SUV"),
    ("jeep", "grand cherokee", "SUV"),  ("nissan", "rogue", "SUV"),
    ("subaru", "outback", "SUV"),       ("hyundai", "tucson", "SUV"),
    ("mazda", "cx-5", "SUV"),           ("kia", "sorento", "SUV"),
    ("ford", "f-150", "Truck"),         ("chevrolet", "silverado", "Truck"),
    ("ram", "1500", "Truck"),           ("toyota", "tacoma", "Truck"),
    ("gmc", "sierra", "Truck"),         ("nissan", "frontier", "Truck"),
    ("tesla", "model 3", "Electric"),   ("tesla", "model y", "Electric"),
    ("tesla", "model s", "Electric"),   ("nissan", "leaf", "Electric"),
    ("chevrolet", "bolt", "Electric"),  ("ford", "mustang mach-e", "Electric"),
    ("toyota", "prius", "Hybrid"),      ("hyundai", "ioniq", "Hybrid"),
    ("honda", "odyssey", "Minivan"),    ("chrysler", "pacifica", "Minivan"),
    ("toyota", "sienna", "Minivan"),
    ("bmw", "3 series", "Luxury"),      ("lexus", "rx", "Luxury"),
    ("audi", "a4", "Luxury"),           ("mercedes-benz", "c-class", "Luxury"),
]


def get_json(url, params=None):
    """GET with caching, retries and backoff. Returns parsed JSON or None."""
    key = hashlib.md5(f"{url}|{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
    blob = CACHE / f"{key}.json"
    if blob.exists():
        try:
            return json.loads(blob.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blob.unlink()

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                data = r.json()
                blob.write_text(json.dumps(data), encoding="utf-8")
                time.sleep(PAUSE)
                return data
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            return None
        except (requests.RequestException, json.JSONDecodeError):
            time.sleep(2 ** attempt)
    return None


def fetch_complaints():
    rows = []
    for make, model, segment in VEHICLES:
        for year in YEARS:
            data = get_json(
                "https://api.nhtsa.gov/complaints/complaintsByVehicle",
                {"make": make, "model": model, "modelYear": year},
            )
            if not data:
                continue
            for rec in data.get("results", []) or []:
                rec = dict(rec)
                rec.pop("products", None)      # nested; make/model/year already known
                rec["query_make"] = make
                rec["query_model"] = model
                rec["query_year"] = year
                rec["segment"] = segment
                rows.append(rec)
        print(f"  complaints  {make} {model}: running total {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def fetch_recalls():
    rows = []
    for make, model, segment in VEHICLES:
        for year in YEARS:
            data = get_json(
                "https://api.nhtsa.gov/recalls/recallsByVehicle",
                {"make": make, "model": model, "modelYear": year},
            )
            if not data:
                continue
            for rec in data.get("results", []) or []:
                rec = dict(rec)
                rec["query_make"] = make
                rec["query_model"] = model
                rec["query_year"] = year
                rec["segment"] = segment
                rows.append(rec)
        print(f"  recalls     {make} {model}: running total {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def fetch_safety_ratings():
    rows = []
    for make, model, segment in VEHICLES:
        for year in YEARS:
            listing = get_json(
                f"https://api.nhtsa.gov/SafetyRatings/modelyear/{year}"
                f"/make/{make}/model/{model}"
            )
            if not listing:
                continue
            for entry in listing.get("Results", []) or []:
                vid = entry.get("VehicleId")
                if not vid:
                    continue
                detail = get_json(f"https://api.nhtsa.gov/SafetyRatings/VehicleId/{vid}")
                if not detail:
                    continue
                for rec in detail.get("Results", []) or []:
                    rec = dict(rec)
                    rec["query_make"] = make
                    rec["query_model"] = model
                    rec["query_year"] = year
                    rec["segment"] = segment
                    rows.append(rec)
        print(f"  ratings     {make} {model}: running total {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def fetch_investigations():
    """Bulk download (not an API): NHTSA ODI investigations flat file."""
    url = "https://static.nhtsa.gov/odi/ffdd/inv/FLAT_INV.zip"
    local_zip = RAW / "FLAT_INV.zip"
    if not local_zip.exists():
        r = requests.get(url, headers=HEADERS, timeout=180)
        r.raise_for_status()
        local_zip.write_bytes(r.content)

    # Field order documented in NHTSA's INV.txt layout file.
    cols = [
        "NHTSA_ACTION_NUMBER", "MAKE", "MODEL", "YEAR", "COMPNAME", "MFR_NAME",
        "ODATE", "CDATE", "CAMPNO", "SUBJECT", "SUMMARY",
    ]
    with zipfile.ZipFile(local_zip) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        with z.open(name) as fh:
            df = pd.read_csv(
                io.TextIOWrapper(fh, encoding="latin-1", errors="replace"),
                sep="\t", header=None, names=cols,
                dtype=str, on_bad_lines="skip", quoting=3,
            )

    # SUMMARY holds multi-kilobyte narratives that inflate the export to
    # hundreds of megabytes. The narrative text is not used by this project's
    # analysis, so it is capped here to keep the raw file distributable.
    df["SUMMARY"] = df["SUMMARY"].str.slice(0, 400)
    return df


# Raw exports are gzipped: the complaint and investigation tables are tens of
# megabytes as plain CSV, which is impractical to host in a git repository.
# pandas infers gzip compression from the .gz extension on both read and write.
STAGES = {
    "complaints":     (fetch_complaints,     "complaints_raw.csv.gz"),
    "recalls":        (fetch_recalls,        "recalls_raw.csv.gz"),
    "ratings":        (fetch_safety_ratings, "safety_ratings_raw.csv.gz"),
    "investigations": (fetch_investigations, "investigations_raw.csv.gz"),
}

if __name__ == "__main__":
    import sys

    # Run one stage at a time ("py code/fetch_data.py complaints") or all of
    # them in sequence. Responses are cached, so any stage is safely re-runnable.
    wanted = sys.argv[1:] or list(STAGES)
    for stage in wanted:
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage!r}; choose from {list(STAGES)}")
        fn, out = STAGES[stage]
        print(f"\n=== {stage} ===", flush=True)
        df = fn()
        df.to_csv(RAW / out, index=False)
        print(f"  -> {out}  {df.shape}", flush=True)
