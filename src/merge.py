"""
Merge the processed yield, climate, vegetation and soil tables into
data/processed/merged.csv.

The crop-mask study reads yield and four soil columns from this file. The
climate and vegetation joins stay because they decide which county-years are
kept: a county-year missing from either table is dropped.

Usage:
    python merge.py
    python merge.py --output some/other/merged.csv
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from config import ANALYSIS_YEARS, DATA_PATH, IRRIGATION_COLUMN, STUDY_CROPS, STUDY_STATES
from utils import save_df

PROCESSED_DIR = DATA_PATH / "processed"
YIELD_FILE = PROCESSED_DIR / "yield.csv"
CLIMATE_FILE = PROCESSED_DIR / "climate.csv"
VEG_FILE = PROCESSED_DIR / "vegetation.csv"
SOIL_FILE = PROCESSED_DIR / "soil.csv"
MERGED_FILE = PROCESSED_DIR / "merged.csv"

# Vegetation's n_months is renamed so it cannot collide with climate's.
VEG_MONTHS_COL = "veg_n_months"


def standardize_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Make county_fips a 5-digit zero-padded string."""
    df = df.copy()
    if "county_fips" in df.columns:
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    return df


def filter_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the analysis years."""
    if "year" not in df.columns:
        return df
    start, end = ANALYSIS_YEARS
    return df[(df["year"] >= start) & (df["year"] <= end)].copy()


def crop_name(row: pd.Series) -> str:
    """Commodity plus class, for example CORN or WHEAT_WINTER.

    The irrigation label is not added, so this needs the aggregate yield table.
    """
    name = str(row.get("commodity_desc", "")).strip().upper()
    crop_class = row.get("class_desc")
    crop_class = "" if pd.isna(crop_class) else str(crop_class).strip().upper()
    if crop_class in ("", "ALL CLASSES"):
        return name
    crop_class = re.sub(r"[^A-Z0-9 ]+", " ", crop_class)
    crop_class = re.sub(r"\s+", "_", crop_class).strip("_")
    return f"{name}_{crop_class}" if crop_class else name


def load_tables():
    """Read the four processed tables and apply the year, crop and state filters."""
    yield_df = filter_by_year(standardize_fips(pd.read_csv(YIELD_FILE)))
    yield_df = yield_df[yield_df["commodity_desc"].isin(STUDY_CROPS)]
    yield_df = yield_df[yield_df["state_alpha"].isin(STUDY_STATES)]
    climate_df = filter_by_year(standardize_fips(pd.read_csv(CLIMATE_FILE)))
    veg_df = filter_by_year(standardize_fips(pd.read_csv(VEG_FILE)))
    soil_df = standardize_fips(pd.read_csv(SOIL_FILE))
    return yield_df, climate_df, veg_df, soil_df


def merge_tables(yield_df, climate_df, veg_df, soil_df) -> pd.DataFrame:
    """Inner-join the four tables on county and year."""
    yield_df = yield_df.copy()
    if IRRIGATION_COLUMN not in yield_df.columns:
        yield_df[IRRIGATION_COLUMN] = "all"
    yield_df["crop"] = yield_df.apply(crop_name, axis=1)
    yield_cols = ["county_fips", "year", "crop", IRRIGATION_COLUMN, "yield_value"]
    yield_df = yield_df[[c for c in yield_cols if c in yield_df.columns]]

    dup = yield_df.duplicated(subset=["county_fips", "year", "crop"]).sum()
    if dup:
        logging.error(f"{dup:,} duplicate (county_fips, year, crop) yield rows")

    climate_exclude = ["county_fips", "year", "county_name", "state_abbr", "state_alpha",
                       "state_fips", "n_months"]
    climate_cols = ["county_fips", "year"] + [c for c in climate_df.columns if c not in climate_exclude]
    climate_df = climate_df[climate_cols].drop_duplicates(subset=["county_fips", "year"])

    veg_exclude = ["county_fips", "year", "county_name", "state_fips", "state_abbr"]
    veg_df = veg_df.rename(columns={"n_months": VEG_MONTHS_COL})
    veg_cols = ["county_fips", "year"] + [c for c in veg_df.columns if c not in veg_exclude]
    veg_df = veg_df[veg_cols].drop_duplicates(subset=["county_fips", "year"])

    soil_exclude = ["county_fips", "state_abbr", "county_name", "state_fips",
                    "mask_start_year", "mask_end_year", "sand_mean"]
    soil_cols = ["county_fips", "state_abbr", "county_name"] + [c for c in soil_df.columns if c not in soil_exclude]
    soil_df = soil_df[[c for c in soil_cols if c in soil_df.columns]].drop_duplicates(subset=["county_fips"])

    merged = yield_df.merge(climate_df, on=["county_fips", "year"], how="inner")
    merged = merged.merge(veg_df, on=["county_fips", "year"], how="inner")
    merged = merged.merge(soil_df, on=["county_fips"], how="inner")
    logging.info(f"Merged: {len(merged):,} rows, {merged['county_fips'].nunique()} counties, "
                 f"{merged['year'].min()}-{merged['year'].max()}")
    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=MERGED_FILE)
    args = ap.parse_args()
    save_df(merge_tables(*load_tables()), args.output)
    logging.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
