"""Download weekly USDA crop progress for the study states.

The file is used to check whether August and September yield predictions occur
before harvest. It downloads state-level corn and soybean progress reports.

Run: python src/download_crop_progress.py
Writes data/raw/crop_progress_weekly.csv by default.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils import redact
from config import NASS_API_KEY, NASS_BASE_URL, DATA_PATH


STATES = ("IA", "IL", "IN", "KS", "MN", "MO", "ND", "NE", "OH", "SD", "WI")
CROPS = ("CORN", "SOYBEANS")
OUTPUT_NAME = "crop_progress_weekly.csv"

log = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def request_rows(params: dict) -> list[dict]:
    response = requests.get(NASS_BASE_URL, params=params, timeout=(15, 300))
    response.raise_for_status()
    return response.json().get("data", [])


def download_state_crop(state: str, crop: str, start_year: int, end_year: int) -> list[dict]:
    rows = []
    offset = 0
    while True:
        params = {
            "key": NASS_API_KEY,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "commodity_desc": crop,
            "statisticcat_desc": "PROGRESS",
            "agg_level_desc": "STATE",
            "state_alpha": state,
            "year__GE": str(start_year),
            "year__LE": str(end_year),
            "freq_desc": "WEEKLY",
            "limit": 50000,
            "offset": offset,
            "format": "JSON",
        }
        page = request_rows(params)
        rows.extend(page)
        if len(page) < 50000:
            return rows
        offset += len(page)
        time.sleep(2)


def clean(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    if data.empty:
        return data

    wanted = data["short_desc"].str.contains(
        "HARVESTED|MATURE|DROPPING LEAVES", case=False, na=False
    )
    data = data[wanted].copy()
    corn_silage = data["short_desc"].str.startswith("CORN, SILAGE", na=False)
    data = data[~corn_silage]
    data["Value"] = pd.to_numeric(
        data["Value"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["week_ending"] = pd.to_datetime(data["week_ending"], errors="coerce")
    data = data[data["Value"].notna() & data["week_ending"].notna()]

    columns = [
        "state_alpha", "year", "week_ending", "commodity_desc", "short_desc",
        "unit_desc", "Value",
    ]
    return data[columns].drop_duplicates().sort_values(
        ["commodity_desc", "state_alpha", "week_ending", "short_desc"]
    )


def validate(data: pd.DataFrame, states: list[str], start_year: int, end_year: int) -> None:
    duplicate_keys = [
        "state_alpha", "year", "week_ending", "commodity_desc", "short_desc"
    ]
    checks = {
        "values_are_percent": data["unit_desc"].str.startswith("PCT").all(),
        "values_are_between_0_and_100": data["Value"].between(0, 100).all(),
        "no_duplicate_reports": not data.duplicated(duplicate_keys).any(),
        "requested_states_present": set(states).issubset(set(data["state_alpha"])),
        "years_in_requested_range": data["year"].between(start_year, end_year).all(),
        "report_date_matches_crop_season": (
            data["week_ending"].dt.year.eq(data["year"])
            | data["week_ending"].dt.year.eq(data["year"] + 1)
        ).all(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Crop progress checks failed: " + ", ".join(failed))
    log.info("Crop progress checks passed: %s", ", ".join(checks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", default=list(STATES))
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directory holding raw/ for this file. Relative "
                             "paths resolve from the repository root. Default "
                             "is data/.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Full output path, overriding --out-dir.")
    parser.add_argument("--log-file", type=Path,
                        default=DATA_PATH.parent / "logs" / "crop_progress.log")
    args = parser.parse_args()
    if args.output is None:
        out_dir = args.out_dir or DATA_PATH
        if not out_dir.is_absolute():
            out_dir = DATA_PATH.parent / out_dir
        args.output = out_dir / "raw" / OUTPUT_NAME

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    if not NASS_API_KEY:
        raise SystemExit("NASS_API_KEY is not set. Add it to the project-root .env file.")

    rows = []
    failures = []
    for crop in CROPS:
        for state in args.states:
            log.info("Downloading %s progress for %s", crop, state)
            try:
                part = download_state_crop(state, crop, args.start_year, args.end_year)
                rows.extend(part)
                log.info("Received %s rows", len(part))
            except requests.RequestException as error:
                failures.append(f"{crop}-{state}")
                log.error("Failed %s %s: %s", crop, state, redact(error))
            time.sleep(1)

    if failures:
        raise SystemExit("Download incomplete: " + ", ".join(failures))

    data = clean(rows)
    if data.empty:
        raise SystemExit("USDA returned no matching crop progress rows")
    data = data[data["year"].between(args.start_year, args.end_year)].copy()
    validate(data, args.states, args.start_year, args.end_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    log.info("Wrote %s rows to %s", len(data), args.output)
    for crop in CROPS:
        labels = sorted(data.loc[data["commodity_desc"] == crop, "short_desc"].unique())
        log.info("%s labels: %s", crop, labels)


if __name__ == "__main__":
    main()
