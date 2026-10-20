"""Check that the downloaded NDVI values match a separate Earth Engine calculation.

Pick three counties per state and recalculate June-September county means for
the generic, same-year crop, and previous-year crop masks. Compare these with
the saved monthly CSVs. This checks NDVI only, not EVI or the extra mask arms.
The calculation is kept separate from the downloader to help catch mistakes.

Run: python validation/verify_masks.py --product MOD13Q1
Writes results/verify_masks<suffix>.csv.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ee
from config import GEE_PROJECT_ID

PRODUCT = {"MOD13A3": "MODIS/061/MOD13A3", "MOD13Q1": "MODIS/061/MOD13Q1"}
SUFFIX = {"MOD13A3": "", "MOD13Q1": "_250m"}
# Repeated here on purpose rather than imported from config.
CORN, SOYBEANS = 1, 5
CROPLAND_RANGES = ((1, 60), (66, 77), (204, 254))
MONTHS = (6, 7, 8, 9)
SEED = 24092026  # Keeps the sampled counties the same each time; any fixed seed works.
COUNTIES_PER_STATE = 3
TOLERANCE = 1e-6


def cdl(year: int):
    return (ee.ImageCollection("USDA/NASS/CDL")
            .filter(ee.Filter.calendarRange(year, year, "year"))
            .first().select("cropland"))


def generic_mask(year: int):
    layer = cdl(year)
    mask = None
    for low, high in CROPLAND_RANGES:
        part = layer.gte(low).And(layer.lte(high))
        mask = part if mask is None else mask.Or(part)
    return mask.selfMask()


def crop_mask(year: int, code: int):
    return cdl(year).eq(code).selfMask()


def month_composites(collection, year: int, month: int, monthly: bool):
    """Images belonging to one month: for a 16-day product, those whose
    window ends inside it."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    if monthly:
        return collection.filterDate(start, end)
    wide = collection.filterDate(start.advance(-1, "month"), end)
    tagged = wide.map(lambda i: i.set(
        "ends", ee.Date(i.get("system:time_start")).advance(15, "day").millis()))
    return tagged.filter(ee.Filter.rangeContains(
        "ends", start.millis(), end.advance(-1, "day").millis()))


def county_means(product: str, year: int, month: int, counties, masks: dict) -> dict:
    collection = ee.ImageCollection(product).select(["NDVI", "SummaryQA"])
    monthly = product.endswith("MOD13A3")
    images = month_composites(collection, year, month, monthly)
    grid = collection.first().select("NDVI").projection()

    def scaled(image):
        return (image.select("NDVI").multiply(0.0001)
                .updateMask(image.select("SummaryQA").lte(1)))

    composite = images.map(scaled).mean().setDefaultProjection(grid)
    bands = [composite.updateMask(m).rename(name) for name, m in masks.items()]
    result = ee.Image.cat(bands).reduceRegions(
        collection=counties, reducer=ee.Reducer.mean(),
        scale=grid.nominalScale(), crs=grid, tileScale=16)
    for attempt in range(3):
        try:
            return result.getInfo()["features"]
        except Exception as error:
            if attempt == 2:
                raise
            print(f"    retry: {error}")
            time.sleep(10 * (attempt + 1))


def compare_saved(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Keep every sampled row, including missing saved records."""
    keys = ["county_fips", "year", "month"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Rebuilt mask sample is empty or contains duplicate keys.")
    reference = reference.merge(frame[keys], on=keys, how="inner", validate="many_to_one")
    return frame.merge(reference[keys + ["ndvi_mean"]], on=keys, how="left",
                       validate="one_to_one", indicator="saved_match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=sorted(PRODUCT), default="MOD13Q1")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    args = parser.parse_args()
    suffix = SUFFIX[args.product]

    saved = {
        "cropland": ROOT / "data" / "raw" / f"vegetation_monthly{suffix}.csv",
        "same_year": ROOT / "data" / "raw" / f"vegetation_monthly_crop_specific{suffix}.csv",
        "previous_year": ROOT / "data" / "raw" / f"vegetation_monthly_crop_specific_prev_year{suffix}.csv",
    }
    missing = [str(p) for p in saved.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing saved files: {missing}")

    ee.Initialize(project=GEE_PROJECT_ID)
    ee.data.setDeadline(300_000)

    generic = pd.read_csv(saved["cropland"], dtype={"county_fips": str, "state_fips": str})
    rng = np.random.default_rng(SEED)
    picked = []
    for state, part in generic.groupby("state_fips"):
        fips = sorted(part.county_fips.unique())
        picked += [(state, f) for f in rng.choice(fips, min(COUNTIES_PER_STATE, len(fips)), replace=False)]
    print(f"{len(picked)} counties, {len(args.years)} years, {len(MONTHS)} months, 3 masks")

    rows = []
    for state in sorted({s for s, _ in picked}):
        fips = [f for s, f in picked if s == state]
        counties = (ee.FeatureCollection("TIGER/2018/Counties")
                    .filter(ee.Filter.eq("STATEFP", state))
                    .map(lambda f: f.set("county_fips", ee.String(f.get("STATEFP")).cat(f.get("COUNTYFP"))))
                    .filter(ee.Filter.inList("county_fips", fips)))
        for year in args.years:
            masks = {
                "cropland": generic_mask(year),
                "same_year_CORN": crop_mask(year, CORN),
                "same_year_SOYBEANS": crop_mask(year, SOYBEANS),
                "previous_year_CORN": crop_mask(year - 1, CORN),
                "previous_year_SOYBEANS": crop_mask(year - 1, SOYBEANS),
            }
            for month in MONTHS:
                start = time.time()
                for feature in county_means(PRODUCT[args.product], year, month, counties, masks):
                    props = feature["properties"]
                    for name in masks:
                        rows.append({"county_fips": props["county_fips"], "year": year,
                                     "month": month, "band": name, "audit_ndvi": props.get(name)})
                print(f"  {state} {year} m{month}: {time.time() - start:.0f} s", flush=True)

    audit = pd.DataFrame(rows)
    expected = {(f, y, m, band) for _, f in picked for y in args.years
                for m in MONTHS for band in (
                    "cropland", "same_year_CORN", "same_year_SOYBEANS",
                    "previous_year_CORN", "previous_year_SOYBEANS",
                )}
    keys = ["county_fips", "year", "month", "band"]
    if audit.empty or audit.duplicated(keys).any() or set(
        audit[keys].itertuples(index=False, name=None)
    ) != expected:
        raise ValueError("Earth Engine returned an incomplete or duplicate sample.")
    parts = []
    for band, frame in audit.groupby("band"):
        if band == "cropland":
            ref = pd.read_csv(saved["cropland"], dtype={"county_fips": str})
        else:
            arm, crop = band.rsplit("_", 1)[0], band.rsplit("_", 1)[1]
            ref = pd.read_csv(saved[arm], dtype={"county_fips": str})
            ref = ref[ref.crop == crop]
        parts.append(compare_saved(frame, ref))
    out = pd.concat(parts, ignore_index=True)
    out["abs_difference"] = (out.audit_ndvi - out.ndvi_mean).abs()
    out["comparable"] = out.abs_difference.notna()
    out["coverage_mismatch"] = (
        out.saved_match.ne("both") | out.audit_ndvi.isna().ne(out.ndvi_mean.isna())
    )
    out["passed"] = out.comparable & (out.abs_difference < TOLERANCE)
    # Report the two CDL grid eras separately; later-year differences need review.
    out["cdl_grid_m"] = out.year.map(lambda y: 30 if y <= 2023 else 10)

    path = ROOT / "results" / f"verify_masks{suffix}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {path}")
    print(f"{len(out)} rows, {int((~out.comparable).sum())} with no value on one side")
    for grid in sorted(out.cdl_grid_m.unique(), reverse=True):
        ok = out[(out.cdl_grid_m == grid) & out.comparable]
        if ok.empty:
            continue
        print(f"  {grid} m CDL: {len(ok)} comparisons, {int(ok.passed.sum())} within "
              f"{TOLERANCE:g}, worst absolute difference {ok.abs_difference.max():.2e}")
    if out.coverage_mismatch.any():
        raise SystemExit("Mask coverage differs or sampled records are missing from the saved inputs.")
    exact = out[(out.cdl_grid_m == 30) & out.comparable]
    if exact.empty:
        raise SystemExit("No comparable 30 m CDL rows; the strict mask check was not performed.")
    if not exact.empty and not exact.passed.all():
        raise SystemExit("Mask application disagrees where CDL is 30 m; that is a real defect.")
    print("Comparable 30 m CDL rows agree within tolerance. The 10 m CDL differences "
          "are reported for review; this check does not establish their cause.")


if __name__ == "__main__":
    main()
