"""Convert monthly vegetation observations into growing-season summaries."""

import pandas as pd
import logging
from pathlib import Path

from config import get_growing_season


def aggregate_to_yearly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly vegetation rows by county, year, and optional crop."""
    if monthly_df.empty:
        return pd.DataFrame()

    df = monthly_df
    if "state_abbr" in df.columns:
        bounds = df["state_abbr"].map(get_growing_season)
        gs_start = pd.Series([bound[0] for bound in bounds], index=df.index)
        gs_end = pd.Series([bound[1] for bound in bounds], index=df.index)
        n_before = len(df)
        df = df[(df["month"] >= gs_start) & (df["month"] < gs_end)]
        if n_before != len(df):
            logging.info(
                "Season filter kept %s of %s monthly rows.",
                f"{len(df):,}",
                f"{n_before:,}",
            )
    else:
        logging.warning(
            "No state_abbr column was found, so all fetched months will be "
            "included in the yearly vegetation summary."
        )

    grouping_cols = ["state_fips", "county_fips", "county_name", "year"]
    if "crop" in df.columns:
        grouping_cols.append("crop")
    if "mask_year" in df.columns:
        grouping_cols.append("mask_year")

    df = df.copy()
    df["_both_present"] = (
        df["ndvi_mean"].notna() & df["evi_mean"].notna()
    ).astype(int)

    return df.groupby(grouping_cols, as_index=False).agg(
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_max_year=("ndvi_mean", "max"),
        evi_mean_year=("evi_mean", "mean"),
        evi_min_year=("evi_mean", "min"),
        evi_max_year=("evi_mean", "max"),
        n_months=("_both_present", "sum"),
        ndvi_std_year=("ndvi_mean", "std"),
        evi_std_year=("evi_mean", "std"),
        ndvi_sum_year=("ndvi_mean", "sum"),
        evi_sum_year=("evi_mean", "sum"),
    )


def rebuild_from_file(monthly_path: Path, yearly_path: Path):
    """Rebuild a yearly table locally, preserving identifiers and all summaries."""
    monthly = pd.read_csv(monthly_path, dtype={"county_fips": str, "state_fips": str})
    yearly = aggregate_to_yearly(monthly)
    if "state_abbr" in monthly and not yearly.empty:
        states = monthly[["state_fips", "state_abbr"]].drop_duplicates()
        yearly = yearly.merge(states, on="state_fips", validate="many_to_one")
    yearly_path.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(yearly_path, index=False)
    logging.info("Wrote %s (%s rows)", yearly_path, len(yearly))
