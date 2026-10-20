"""Check that the model inputs have sensible values and no missing entries.

This checks climate, soil, and vegetation for the three main masks, plus
corn and soybean yield units. A value can pass these ranges even if a unit
conversion was wrong, so the source conversions still need to be reviewed.

Run: python validation/check_units.py --product MOD13Q1
Writes results/unit_checks<suffix>.csv and .md.
"""

from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import compare_masks_by_month as masks


DATA = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, low_memory=False)


def add_range_check(
    rows: list[dict],
    group: str,
    frame: pd.DataFrame,
    column: str,
    source_unit: str,
    model_unit: str,
    low: float,
    high: float,
    note: str,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    observed = values.dropna()
    minimum = float(observed.min())
    maximum = float(observed.max())
    missing = int(values.isna().sum())
    passed = len(observed) > 0 and missing == 0 and minimum >= low and maximum <= high
    rows.append({
        "group": group,
        "variable": column,
        "source_unit": source_unit,
        "model_unit": model_unit,
        "minimum": minimum,
        "maximum": maximum,
        "missing": missing,
        "passed": passed,
        "note": note,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=sorted(masks.PRODUCT_SUFFIX), default="MOD13Q1")
    product = parser.parse_args().product
    suffix = masks.select_product(product)

    merged = pd.read_csv(DATA / "merged.csv", low_memory=False)
    yield_data = read_csv("yield.csv")

    model_crops = merged[merged["crop"].isin(["CORN", "SOYBEANS"])].copy()
    model_crops["county_fips"] = model_crops["county_fips"].astype(str).str.zfill(5)
    yield_crops = yield_data[yield_data["commodity_desc"].isin(["CORN", "SOYBEANS"])].copy()
    vegetation_inputs, _ = masks.audit_inputs()
    # Only the three main arms, so the checked rows are the modeled rows. The
    # rotation arm has a smaller sample and would drop corn county-years.
    vegetation_inputs = {arm: vegetation_inputs[arm] for arm in masks.BASE_ARMS}
    climate_monthly = masks.read_csv(masks.CLIMATE_MONTHLY)
    base = model_crops[
        ["crop", "state_abbr", "county_fips", "year", "yield_value"]
        + masks.SOIL_FEATURES
    ].drop_duplicates(["crop", "county_fips", "year"])
    cutoff_tables = [
        masks.build_cutoff_table(base, climate_monthly, vegetation_inputs, month)
        for month in masks.CUTOFFS
    ]
    inputs = pd.concat(cutoff_tables, ignore_index=True)
    if inputs.empty:
        raise ValueError("No complete cutoff inputs are available to check.")
    rows: list[dict] = []

    climate_checks = {
        "TMIN": ("K", "degrees C", -60, 60, "Daily minimum temperature, converted from K before monthly aggregation."),
        "TMAX": ("K", "degrees C", -60, 60, "Daily maximum temperature, converted from K before monthly aggregation."),
        "PRCP": ("mm/day", "mm/day", 0, 100, "Daily precipitation totals summed, then divided by observed season days."),
        "VPD": ("kPa", "kPa", 0, 10, "Growing-season day-weighted mean vapor pressure deficit."),
        "ETO": ("mm/day", "mm/day", 0, 20, "Daily reference ET summed, then divided by observed season days."),
        "SRAD": ("W/m2", "W/m2", 0, 600, "Growing-season day-weighted mean shortwave radiation."),
        "GDD_TMAX": ("degrees C/day", "degrees C/day", 0, 30, "Tmax-based heat exposure rate, not conventional GDD."),
        "EDD_TMAX": ("degrees C/day", "degrees C/day", 0, 30, "Tmax degrees above 29 C per season day."),
        "HOT_DAY_FRAC": ("fraction", "fraction", 0, 1, "Fraction of season days with Tmax above 29 C."),
        "TMAX_MAX": ("degrees C", "degrees C", -60, 60, "Maximum daily Tmax during the included months."),
    }
    for column, details in climate_checks.items():
        add_range_check(rows, "climate", inputs, column, *details)

    soil_checks = {
        "clay_mean": ("percent kg/kg", "percent kg/kg", 2, 100, "Thickness-weighted 0-30 cm OpenLandMap estimate."),
        "ph_mean": ("pH x 10", "pH", 4.2, 11, "OpenLandMap values multiplied by 0.1."),
        "soc_mean": ("stored unit x 5 g/kg", "g/kg", 0, 600, "OpenLandMap values multiplied by 5."),
        "bdod_mean": ("stored unit x 10 kg/m3", "g/cm3", 0.05, 1.85, "OpenLandMap values multiplied by 0.01."),
    }
    for column, details in soil_checks.items():
        add_range_check(rows, "soil", inputs, column, *details)

    for arm in masks.BASE_ARMS:
        for column in masks.VEG_FEATURES:
            model_column = f"{column}__{arm}"
            add_range_check(
                rows,
                "vegetation",
                inputs,
                model_column,
                "integer x 0.0001",
                "unitless index",
                -0.2,
                1.0,
                f"{product}, {arm}: expected MODIS scale factor 0.0001.",
            )

    units = sorted(yield_crops["yield_unit"].dropna().unique())
    yield_values = pd.to_numeric(yield_crops["yield_value"], errors="coerce")
    rows.append({
        "group": "yield",
        "variable": "yield_value",
        "source_unit": ", ".join(units),
        "model_unit": "BU / ACRE",
        "minimum": float(yield_values.min()),
        "maximum": float(yield_values.max()),
        "missing": int(yield_values.isna().sum()),
        "passed": units == ["BU / ACRE"] and yield_values.notna().all() and (yield_values > 0).all(),
        "note": "Corn and soybean are modeled separately; both use bushels per acre.",
    })

    temp_order_ok = bool((inputs["TMIN"] <= inputs["TMAX"]).all())
    rows.append({
        "group": "cross_check",
        "variable": "TMIN_le_TMAX",
        "source_unit": "degrees C",
        "model_unit": "degrees C",
        "minimum": "",
        "maximum": "",
        "missing": 0,
        "passed": temp_order_ok,
        "note": "Every modeled county-year must have TMIN no greater than TMAX.",
    })

    audit = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_DIR / f"unit_checks{suffix}.csv", index=False)

    lines = [
        "# Data Unit Audit",
        "",
        "Climate, soil, and all three vegetation arms are checked on complete June-September cutoff inputs.",
        "Yield units and positive values are checked separately in the source corn/soybean table.",
        "These checks detect implausible values; they do not independently verify conversions.",
        "",
        "| Group | Variable | Model unit | Minimum | Maximum | Missing | Pass |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['group']} | {row['variable']} | {row['model_unit']} | "
            f"{row['minimum']} | {row['maximum']} | {row['missing']} | {row['passed']} |"
        )
    lines.extend([
        "",
        "## Important Transformations",
        "",
        "- `PRCP` and `ETO` are millimeters per season day, not seasonal totals.",
        "- `GDD_TMAX` and `EDD_TMAX` are Tmax-based degrees C per season day.",
        "- NDVI and EVI are unitless after applying the MODIS 0.0001 scale factor.",
        "- Corn and soybean yield is in bushels per acre and the crops are modeled separately.",
        "- Soil values are static county summaries over a stable cropland mask.",
    ])
    (OUT_DIR / f"unit_checks{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = audit[~audit["passed"]]
    if not failed.empty:
        raise ValueError(f"Unit audit failed:\n{failed.to_string(index=False)}")
    print(f"All {len(audit)} unit and range checks passed.")


if __name__ == "__main__":
    main()
