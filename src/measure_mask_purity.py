"""Measure how much of each mask's vegetation signal is the target crop.

A MODIS pixel is much larger than a CDL pixel, so a masked pixel is only
partly the target crop. For every pixel a mask keeps, this measures the share
of its area the CDL calls the target crop, averaged over the county the same
way the county mean is. It uses the August composite, and only where the
quality filter kept an observation.

Run: python src/measure_mask_purity.py --product MOD13Q1 --all-counties

Writes results/mask_purity<suffix>.csv and its _summary file.
--summarize-only rebuilds the summary from the saved measurements without
Earth Engine.
"""

import argparse
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from config import GEE_PROJECT_ID, get_cdl_codes
from utils import get_crop_mask, get_crop_type_mask, get_tiger_counties_fc, native_scale

ROOT = Path(__file__).resolve().parents[1]

PRODUCT = {"MOD13A3": "MODIS/061/MOD13A3", "MOD13Q1": "MODIS/061/MOD13Q1"}
SUFFIX = {"MOD13A3": "", "MOD13Q1": "_250m"}
YEARS = (2022, 2023, 2024, 2025)
MONTH = 8
CROPS = ("CORN", "SOYBEANS")
ARMS = ("cropland", "same_year_oracle", "previous_year")
SEED = 20260923
PER_STATE = 10
# Pass rule, fixed before the first run. See protocol_250m.md.
MIN_SHARE = 0.50
MIN_GAP = 0.15


class Resumable:
    """A CSV written as work completes, so a failed run repeats only the last
    unit instead of everything. Keyed by the columns in `keys`."""

    def __init__(self, path: Path, keys: list[str]) -> None:
        self.path = Path(path)
        self.keys = keys
        self._done = self._read_done()

    def _read_done(self) -> set[tuple]:
        if not self.path.exists():
            return set()
        saved = pd.read_csv(self.path, usecols=self.keys, dtype=str)
        return set(map(tuple, saved.values))

    def is_done(self, *values) -> bool:
        return tuple(str(v) for v in values) in self._done

    @property
    def finished(self) -> int:
        return len(self._done)

    def add(self, rows: list[dict]) -> None:
        """Append one unit of work and mark it finished."""
        if not rows:
            return
        frame = pd.DataFrame(rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        self._done |= set(map(tuple, frame[self.keys].astype(str).values))

    def frame(self, dtype: dict | None = None) -> pd.DataFrame:
        """Everything finished so far, read back from disk."""
        if not self.path.exists():
            return pd.DataFrame()
        rows = pd.read_csv(self.path, dtype=dtype)
        duplicated = rows.duplicated(self.keys)
        if duplicated.any():
            raise ValueError(
                f"{self.path} has {int(duplicated.sum())} duplicate rows for "
                f"{self.keys}; delete it and rerun rather than trusting it."
            )
        return rows

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
        self._done = set()


def eligible_counties(all_counties: bool, features: Path) -> pd.DataFrame:
    """Counties with every test year present, either all of them or a sample."""
    table = pd.read_csv(features, usecols=["cutoff_month", "crop", "state_abbr",
                                           "county_fips", "year"],
                        dtype={"county_fips": str})
    table = table[(table.cutoff_month == MONTH) & table.year.isin(YEARS)]
    rng = np.random.default_rng(SEED)
    rows = []
    for crop in CROPS:
        for state in sorted(table.state_abbr.unique()):
            part = table[(table.crop == crop) & (table.state_abbr == state)]
            complete = part.groupby("county_fips").year.nunique()
            keep = sorted(complete[complete == len(YEARS)].index)
            if not all_counties:
                keep = sorted(rng.choice(keep, min(PER_STATE, len(keep)), replace=False))
            rows += [{"crop": crop, "state_abbr": state, "county_fips": f} for f in keep]
    return pd.DataFrame(rows)


def cdl(year: int):
    return (ee.ImageCollection("USDA/NASS/CDL")
            .filter(ee.Filter.calendarRange(year, year, "year"))
            .first().select("cropland"))


def crop_fraction(year: int, crop: str, quality_ok):
    """Share of each MODIS pixel that CDL calls the target crop."""
    codes = get_cdl_codes(crop)
    target = cdl(year).eq(codes[0])
    for code in codes[1:]:
        target = target.Or(cdl(year).eq(code))
    # Earth Engine cannot reduce 10 m CDL (2024 on) to the MODIS grid in one
    # step, so average onto CDL's own 30 m grid first. A 30 m pixel is exactly
    # 3x3 of the 10 m ones, and for the 30 m years this step changes nothing.
    on_30m = (target.reduceResolution(ee.Reducer.mean(), maxPixels=64)
              .reproject(cdl(year).projection().atScale(30)))
    return on_30m.reduceResolution(ee.Reducer.mean(), maxPixels=65535).updateMask(quality_ok)


def measure(product: str, year: int, crop: str, counties) -> list[dict]:
    start = ee.Date.fromYMD(year, MONTH, 1)
    source = (ee.ImageCollection(product).select(["NDVI", "SummaryQA"])
              .filterDate(start, start.advance(1, "month")).first())
    quality_ok = source.select("SummaryQA").lte(1)
    fraction = crop_fraction(year, crop, quality_ok)
    weight = fraction.multiply(0).add(1)

    masks = {
        "cropland": get_crop_mask(year),
        "same_year_oracle": get_crop_type_mask(year, crop),
        "previous_year": get_crop_type_mask(year - 1, crop),
    }
    bands = []
    for arm, mask in masks.items():
        bands.append(fraction.updateMask(mask).rename(f"share__{arm}"))
        bands.append(weight.updateMask(mask).rename(f"pixels__{arm}"))

    grid = source.select("NDVI").projection()
    reducer = ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True)
    features = ee.Image.cat(bands).reduceRegions(
        collection=counties, reducer=reducer,
        scale=native_scale(product), crs=grid, tileScale=16)
    for attempt in range(3):
        try:
            got = features.getInfo()["features"]
            break
        except Exception as error:
            if attempt == 2:
                raise
            print(f"    retry: {error}")
            time.sleep(10 * (attempt + 1))
    return [{"crop": crop, "county_fips": f["properties"]["county_fips"], "year": year,
             "arm": arm, "target_share": f["properties"].get(f"share__{arm}_mean"),
             "pixels": f["properties"].get(f"pixels__{arm}_sum")}
            for f in got for arm in ARMS]


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows, lines = [], []
    for crop in CROPS:
        part = detail[detail.crop == crop]
        pooled = part.groupby("arm").target_share.mean()
        counts = part.groupby("arm").target_share.count()
        by_state = part.groupby(["state_abbr", "arm"]).target_share.mean().unstack()
        state_counts = part.groupby(["state_abbr", "arm"]).target_share.count().unstack()
        for arm in ARMS:
            rows.append({"crop": crop, "level": "pooled", "state_abbr": "ALL", "arm": arm,
                         "mean_share": pooled[arm], "n_county_years": counts[arm]})
        for state, values in by_state.iterrows():
            for arm in ARMS:
                rows.append({"crop": crop, "level": "state", "state_abbr": state, "arm": arm,
                             "mean_share": values[arm],
                             "n_county_years": state_counts.loc[state, arm]})
        same, generic = pooled["same_year_oracle"], pooled["cropland"]
        wrong_way = sorted(by_state.index[by_state.same_year_oracle <= by_state.cropland])
        verdict = "PASS" if (same >= MIN_SHARE and same - generic >= MIN_GAP
                             and not wrong_way) else "FAIL"
        lines += [
            f"{crop}: {verdict}",
            f"  same-year share {same:.3f} (needs >= {MIN_SHARE})",
            f"  generic cropland {generic:.3f}, gap {same - generic:.3f} (needs >= {MIN_GAP})",
            f"  previous-year {pooled['previous_year']:.3f} (descriptive)",
            f"  states where the same-year mask does not beat cropland: {wrong_way or 'none'}",
        ]
    return pd.DataFrame(rows), lines


def write_summaries(detail: pd.DataFrame, suffix: str) -> None:
    summary, lines = summarize(detail)
    summary_path = ROOT / "results" / f"mask_purity{suffix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=sorted(PRODUCT), default="MOD13Q1")
    parser.add_argument("--all-counties", action="store_true",
                        help="Use every eligible county instead of a fixed sample "
                             "of ten per state.")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Rebuild the summary files from the saved measurements.")
    args = parser.parse_args()
    suffix = SUFFIX[args.product] + ("_allcounties" if args.all_counties else "")

    detail_path = ROOT / "results" / f"mask_purity{suffix}.csv"
    if args.summarize_only:
        if not detail_path.exists():
            raise SystemExit(f"Need {detail_path}; measure it first.")
        write_summaries(pd.read_csv(detail_path, dtype={"county_fips": str}), suffix)
        return
    if detail_path.exists():
        raise SystemExit(f"Refusing to overwrite {detail_path}")
    # Each crop, state and year is written as it finishes, so a dropped
    # connection or an exhausted quota costs only the piece in flight.
    partial = Resumable(detail_path.with_name(detail_path.stem + "_partial.csv"),
                        ["crop", "county_fips", "year", "arm"])

    features = ROOT / "data" / "processed" / f"features_by_cutoff{SUFFIX[args.product]}.csv"
    if not features.exists():
        raise SystemExit(f"Need {features}; build it with compare_masks_by_month.py --table-only.")

    ee.Initialize(project=GEE_PROJECT_ID)
    ee.data.setDeadline(300_000)

    sample = eligible_counties(args.all_counties, features)
    sample["state_fips"] = sample.county_fips.str[:2]
    print(f"{sample.county_fips.nunique()} counties, {len(YEARS)} years, "
          f"{len(CROPS)} crops, month {MONTH}")

    if partial.finished:
        print(f"resuming: {partial.finished} rows already measured")
    groups = sample.groupby(["crop", "state_abbr", "state_fips"])
    for number, ((crop, state, state_fips), part) in enumerate(groups, start=1):
        counties = None
        for year in YEARS:
            if partial.is_done(crop, part.county_fips.iloc[0], year, ARMS[0]):
                continue
            if counties is None:
                counties = get_tiger_counties_fc(state_fips).filter(
                    ee.Filter.inList("county_fips", part.county_fips.tolist()))
            start = time.time()
            partial.add(measure(PRODUCT[args.product], year, crop, counties))
            print(f"[{number}/{len(groups)}] {crop} {state} {year}: "
                  f"{time.time() - start:.0f} s", flush=True)

    detail = partial.frame(dtype={"county_fips": str}).merge(
        sample[["crop", "county_fips", "state_abbr"]], on=["crop", "county_fips"])
    detail.to_csv(detail_path, index=False)
    print(f"\nWrote {detail_path}")
    partial.clear()
    write_summaries(detail, suffix)


if __name__ == "__main__":
    main()
