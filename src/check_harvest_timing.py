"""Check whether crop-mask gains occur before harvest.

Refits August/September models and joins county-year errors to weekly crop
progress. County-year errors are saved only when --save-errors is supplied.

Run: python src/check_harvest_timing.py
Writes results/harvest_timing<suffix>.csv.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import compare_masks_by_month as masks

PROGRESS_FILE = masks.DATA / "raw/crop_progress_weekly.csv"
SUMMARY_OUTPUT = masks.OUT_DIR / "harvest_timing.csv"
MONTHS = (8, 9)
ARMS = ("cropland", "same_year_oracle")
THRESHOLDS = (0, 5, 10)

log = logging.getLogger(__name__)


def stage_name(description: str):
    desc = description.upper()
    if "HARVESTED" in desc:
        return "harvested"
    if "DROPPING LEAVES" in desc:
        return "dropping_leaves"
    if "MATURE" in desc:
        return "mature"
    return None


def read_progress():
    df = pd.read_csv(PROGRESS_FILE, parse_dates=["week_ending"])
    df = df[~df["short_desc"].str.startswith("CORN, SILAGE", na=False)].copy()
    df["stage"] = df["short_desc"].map(stage_name)
    df = df[df["stage"].notna()]
    df["crop"] = df["commodity_desc"].map({"CORN": "CORN", "SOYBEANS": "SOYBEANS"})
    return df


def progress_at_cutoff(progress, crop, state, year, stage, month):
    subset = progress[
                (progress["crop"] == crop) &
                (progress["state_alpha"] == state) &
                (progress["year"] == year) &
                (progress["stage"] == stage)
            ].sort_values("week_ending")

    if subset.empty:
        return np.nan, "", False

    cutoff_date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    cutoff_data = subset[subset["week_ending"] <= cutoff_date]

    if not cutoff_data.empty:
        latest = cutoff_data.iloc[-1]
        return float(latest["Value"]), latest["week_ending"].date().isoformat(), False

    # Reporting started after cutoff; zero means unrecorded, not zero actual progress
    return 0.0, cutoff_date.date().isoformat(), True


def add_progress(errors, progress):
    records = []
    keys = errors[["crop", "state", "year", "cutoff_month"]].drop_duplicates()

    for crop, state, year, cutoff_month in keys.itertuples(index=False):
        harvested, h_date, h_inferred = progress_at_cutoff(
            progress, crop, state, year, "harvested", cutoff_month
        )

        dev_stage = "mature" if crop == "CORN" else "dropping_leaves"
        developed, d_date, d_inferred = progress_at_cutoff(
            progress, crop, state, year, dev_stage, cutoff_month
        )

        records.append({
            "crop": crop,
            "state": state,
            "year": year,
            "cutoff_month": cutoff_month,
            "harvested_pct": harvested,
            "harvest_report_date": h_date,
            "harvest_zero_inferred": h_inferred,
            "development_stage": dev_stage,
            "development_pct": developed,
            "development_report_date": d_date,
            "development_zero_inferred": d_inferred,
        })

    return errors.merge(
        pd.DataFrame(records),
        on=["crop", "state", "year", "cutoff_month"],
        how="left",
        validate="many_to_one",
    )


def model_errors():
    vegetation, _ = masks.audit_inputs()
    vegetation = {arm: vegetation[arm] for arm in masks.BASE_ARMS}
    climate = masks.read_csv(masks.CLIMATE_MONTHLY)
    base = masks.load_base_table()

    records = []
    for month in MONTHS:
        table = masks.build_cutoff_table(base, climate, vegetation, month)

        for crop in masks.CROPS:
            df_crop = table[table["crop"] == crop].copy()
            train, test, _ = masks.temporal_split(df_crop)
            actuals = test["yield_value"].to_numpy()

            for model_name in masks.MODELS:
                preds = {}
                for arm in ARMS:
                    log.info("%s month=%s model=%s arm=%s", crop, month, model_name, arm)
                    preds[arm] = masks.predict(
                        train, test, masks.feature_columns(arm), masks.SEEDS, model_name
                    )

                for i, r in enumerate(test.itertuples(index=False)):
                    y_true = actuals[i]
                    p_crop = preds["cropland"][i]
                    p_oracle = preds["same_year_oracle"][i]

                    records.append({
                        "crop": crop,
                        "cutoff_month": month,
                        "model": model_name,
                        "state": r.state_abbr,
                        "county_fips": r.county_fips,
                        "year": r.year,
                        "actual_yield": y_true,
                        "cropland_prediction": p_crop,
                        "same_year_prediction": p_oracle,
                        "cropland_squared_error": (y_true - p_crop) ** 2,
                        "same_year_squared_error": (y_true - p_oracle) ** 2,
                    })

    return pd.DataFrame(records)


def summarize(data):
    records = []

    for (crop, month, model_name), group in data.groupby(["crop", "cutoff_month", "model"]):
        filters = [("all_with_progress", group["harvested_pct"].notna())]
        filters.extend([
            (f"harvested_at_most_{t}_pct", group["harvested_pct"] <= t)
            for t in THRESHOLDS
        ])

        for sample_name, mask in filters:
            sub = group[mask]
            if sub.empty:
                continue

            ref_err = sub["cropland_squared_error"].to_numpy()
            mask_err = sub["same_year_squared_error"].to_numpy()

            q = float(np.sqrt(mask_err.mean() / ref_err.mean()))
            lo, hi = masks.bootstrap_q(
                mask_err, ref_err, sub["county_fips"].to_numpy(), masks.BOOTSTRAP_DRAWS
            )

            records.append({
                "crop": crop,
                "cutoff_month": month,
                "model": model_name,
                "sample": sample_name,
                "n": len(sub),
                "n_state_years": sub[["state", "year"]].drop_duplicates().shape[0],
                "mean_harvested_pct": sub["harvested_pct"].mean(),
                "maximum_harvested_pct": sub["harvested_pct"].max(),
                "mean_development_pct": sub["development_pct"].mean(),
                "q_vs_cropland": q,
                "q_vs_cropland_lo": lo,
                "q_vs_cropland_hi": hi,
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-errors",
        type=Path,
        default=None,
        help="Optional path for saving county-year error table.",
    )
    parser.add_argument("--product", choices=sorted(masks.PRODUCT_SUFFIX), default="MOD13Q1")
    args = parser.parse_args()

    suffix = masks.select_product(args.product)
    out_file = SUMMARY_OUTPUT.with_name(f"harvest_timing{suffix}.csv")

    log_dir = masks.ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "harvest_timing.log", encoding="utf-8")],
    )

    progress = read_progress()
    errors = add_progress(model_errors(), progress)
    summary = summarize(errors)

    if args.save_errors:
        args.save_errors.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(args.save_errors, index=False)
        log.info("Wrote %s", args.save_errors)

    summary.to_csv(out_file, index=False)
    log.info("Wrote %s", out_file)


if __name__ == "__main__":
    main()
