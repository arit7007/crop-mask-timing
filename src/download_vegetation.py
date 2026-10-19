"""Download county NDVI and EVI from MODIS with Google Earth Engine.

Applies a CDL crop mask, uses state growing seasons, and makes one
reduceRegions call per state-year. MOD13A3 (monthly, 1 km) is the default;
--product MOD13Q1 gives 16-day 250 m and needs --variant.

Writes raw/vegetation_monthly.csv and processed/vegetation.csv.

    python download_vegetation.py
    python download_vegetation.py --states TX CA --start_year 2015 --end_year 2020
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

try:
    import ee
except ImportError:
    ee = None

from config import (
    DATA_PATH,
    DATA_PATH_RAW as RAW_DIR,
    GEE_PROJECT_ID,
    DEFAULT_STATES,
    ANALYSIS_YEARS,
    CROP_SPECIFIC_MASK_CROPS,
    get_growing_season,
    get_required_months,
    get_cdl_codes,
    resolve_states,
)
from utils import (
    native_scale,
    fetch_state_fips,
    get_crop_mask,
    get_crop_type_mask,
    get_tiger_counties_fc,
    logging,
)
from vegetation_aggregation import aggregate_to_yearly, rebuild_from_file

RETRY_LOG = logging.getLogger(__name__)
CODE_ERROR_TYPES = (NameError, AttributeError, ImportError, TypeError, KeyError, ValueError)

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------
PROCESSED_DIR = DATA_PATH / "processed"

MONTHLY_OUTPUT_FILE = RAW_DIR / "vegetation_monthly.csv"
YEARLY_OUTPUT_FILE = PROCESSED_DIR / "vegetation.csv"
CHECKPOINT_FILE = RAW_DIR / "vegetation_checkpoint.json"

# Nondefault masks, products and scales use separate variant files.
USE_CROP_MASK = True
REDUCE_SCALE = None          # None means the asset's native scale
# MOD13A3 is monthly at 1 km. MOD13Q1 is 16-day at 250 m; its composites are
# combined by month in get_monthly_vegetation_stats().
MODIS_PRODUCTS = {
    "MOD13A3": "MODIS/061/MOD13A3",
    "MOD13Q1": "MODIS/061/MOD13Q1",
}
MODIS_PRODUCT = MODIS_PRODUCTS["MOD13A3"]
CROP_SPECIFIC_MASK = False
MASK_CROPS: List[str] = list(CROP_SPECIFIC_MASK_CROPS)
CURRENT_MASK_CROP: Optional[str] = None
MASK_YEAR_OFFSET = 0


# ---------------------------------------------------------------------------
# GEE INITIALIZATION
# ---------------------------------------------------------------------------

def authenticate_gee(project_id: str = GEE_PROJECT_ID) -> None:
    """Initialize Google Earth Engine."""
    if ee is None:
        raise RuntimeError(
            "earthengine-api is required to download vegetation data. "
            "Install it with: pip install earthengine-api"
        )
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    # Without a deadline a dropped connection waits forever and the retry
    # below never fires, which stalls a long run instead of failing it.
    ee.data.setDeadline(300_000)
    logging.info("Google Earth Engine authenticated.")


def setup_file_logging(log_file: Optional[str]) -> None:
    """Write progress messages to a file when a path is provided."""
    if not log_file:
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    resolved = str(path.resolve())
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved:
            return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(handler)
    logging.info(f"Logging to file: {path}")


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGEMENT
# ---------------------------------------------------------------------------

def load_checkpoint() -> Dict:
    """Load checkpoint file tracking completed state-years."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": []}


def save_checkpoint(checkpoint: Dict) -> None:
    """Save checkpoint to file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def _key(state_fips: str, year: int, crop_name: Optional[str] = None) -> str:
    if crop_name:
        return f"{state_fips}_{year}_{crop_name}"
    return f"{state_fips}_{year}"


def is_completed(checkpoint: Dict, state_fips: str, year: int,
                 crop_name: Optional[str] = None) -> bool:
    """Check if a state-year is already completed."""
    return _key(state_fips, year, crop_name) in checkpoint.get("completed", [])


def mark_completed(checkpoint: Dict, state_fips: str, year: int,
                   crop_name: Optional[str] = None) -> None:
    """Mark a state-year as completed."""
    k = _key(state_fips, year, crop_name)
    if k not in checkpoint["completed"]:
        checkpoint["completed"].append(k)
    save_checkpoint(checkpoint)


# ---------------------------------------------------------------------------
# SMALL I/O HELPERS
# ---------------------------------------------------------------------------

def append_csv(df: pd.DataFrame, path) -> None:
    """Append dataframe to CSV (create with header if not exists)."""
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def is_retryable_exception(exc: BaseException) -> bool:
    """Retry temporary Earth Engine or network errors, but not code errors."""
    return not isinstance(exc, CODE_ERROR_TYPES)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception(is_retryable_exception),
    before_sleep=before_sleep_log(RETRY_LOG, logging.WARNING),
    reraise=True,
)
def get_monthly_vegetation_stats_with_retry(
    state_fips: str,
    state_abbr: str,
    year: int,
    crop_name: Optional[str] = None,
) -> pd.DataFrame:
    return get_monthly_vegetation_stats(
        state_fips=state_fips,
        state_abbr=state_abbr,
        year=year,
        crop_name=crop_name,
    )


# ---------------------------------------------------------------------------
# CACHES (avoid rebuilding same GEE objects repeatedly)
# ---------------------------------------------------------------------------

_CROP_MASK_CACHE: Dict[int, ee.Image] = {}
_CROP_TYPE_MASK_CACHE: Dict[tuple[int, str], ee.Image] = {}
_COUNTIES_CACHE: Dict[str, ee.FeatureCollection] = {}

def crop_mask_cached(year: int, crop_name: Optional[str] = None) -> ee.Image:
    if crop_name:
        key = (year, crop_name)
        if key not in _CROP_TYPE_MASK_CACHE:
            _CROP_TYPE_MASK_CACHE[key] = get_crop_type_mask(year, crop_name)
        return _CROP_TYPE_MASK_CACHE[key]
    if year not in _CROP_MASK_CACHE:
        _CROP_MASK_CACHE[year] = get_crop_mask(year)
    return _CROP_MASK_CACHE[year]

def counties_cached(state_fips: str) -> ee.FeatureCollection:
    if state_fips not in _COUNTIES_CACHE:
        _COUNTIES_CACHE[state_fips] = get_tiger_counties_fc(state_fips)
    return _COUNTIES_CACHE[state_fips]


# ---------------------------------------------------------------------------
# CORE GEE PROCESSING (FAST: ONE reduceRegions PER STATE-YEAR)
# ---------------------------------------------------------------------------

def get_monthly_vegetation_stats(
    state_fips: str,
    state_abbr: str,
    year: int,
    crop_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download monthly vegetation statistics with one county reduction per state-year.

    Returns monthly rows for the configured window, including season margins.

    Columns:
      state_fips, county_fips, county_name, year, month, ndvi_mean, evi_mean
    """
    # Keep extra months so season boundaries can be changed locally.
    months_to_fetch = get_required_months(slack=1)

    modis = ee.ImageCollection(MODIS_PRODUCT).select(["NDVI", "EVI", "SummaryQA"])

    # The offset applies to the generic cropland mask as well, so a
    # previous-year generic baseline can be built for comparison against the
    # previous-year crop-specific mask.
    mask_year = year + MASK_YEAR_OFFSET
    if mask_year < 2008:
        raise ValueError(
            f"CDL starts in 2008; cannot build a {mask_year} mask for "
            f"vegetation year {year} with mask-year offset {MASK_YEAR_OFFSET}."
        )
    crop_mask = crop_mask_cached(mask_year, crop_name)
    counties_fc = counties_cached(state_fips)

    grid = modis.first().select("NDVI").projection()

    def scaled(src):
        # SummaryQA: 0 good, 1 marginal, 2 snow/ice, 3 cloud. Keep 0-1 only.
        return (
            src.select(["NDVI", "EVI"])
            .multiply(0.0001)   # MODIS scale factor
            .updateMask(src.select("SummaryQA").lte(1))
        )

    def month_images(start, end):
        if MODIS_PRODUCT == MODIS_PRODUCTS["MOD13A3"]:
            # Already monthly, one image per month, labelled by its start.
            # Its system:time_end is not a reliable cover date: the February
            # 2023 image reports March 3.
            return modis.filterDate(start, end)
        # 16-day product. Keep only composites whose whole window ends within
        # the month, so no observation after the cutoff reaches its features.
        # The window is 16 days from the start date.
        window = modis.filterDate(start.advance(-1, "month"), end)
        return window.map(
            lambda src: src.set(
                "cover_end", ee.Date(src.get("system:time_start")).advance(15, "day").millis()
            )
        ).filter(ee.Filter.rangeContains(
            "cover_end", start.millis(), end.advance(-1, "day").millis()
        ))

    def month_to_img(m):
        m = ee.Number(m).toInt()
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")
        in_month = month_images(start, end)

        # mean() returns an image with no fixed projection, so put the
        # product's own grid back before the crop mask is applied. Masking a
        # gridless image samples the CDL at the wrong resolution.
        img = in_month.map(scaled).mean().setDefaultProjection(grid)
        if USE_CROP_MASK:
            img = img.updateMask(crop_mask)

        mm = m.format("%02d")
        return img.rename([ee.String("NDVI_").cat(mm), ee.String("EVI_").cat(mm)])

    months = ee.List(months_to_fetch)

    # Stack monthly images into one multiband image
    stacked = ee.ImageCollection(months.map(month_to_img)).toBands()

    # toBands prefixes band names like "0_NDVI_04". Remove the numeric prefix.
    band_names = stacked.bandNames()
    cleaned = band_names.map(lambda b: ee.String(b).replace('^[0-9]+_', ''))
    stacked = stacked.rename(cleaned)

    # Reduce once across counties
    fc = stacked.reduceRegions(
        collection=counties_fc,
        reducer=ee.Reducer.mean(),
        scale=REDUCE_SCALE or native_scale(MODIS_PRODUCT),
        tileScale=4,
    )

    result = fc.getInfo()

    rows = []
    for f in result.get("features", []):
        props = f.get("properties", {})

        base = {
            "state_fips": str(props.get("state_fips", "")),
            "county_fips": str(props.get("county_fips", "")),
            "county_name": props.get("county_name", ""),
            "year": int(year),
        }
        if crop_name:
            base["crop"] = crop_name
            base["mask_year"] = int(mask_year)

        for month in months_to_fetch:
            mm = f"{month:02d}"
            rows.append({
                **base,
                "month": int(month),
                "ndvi_mean": props.get(f"NDVI_{mm}"),
                "evi_mean": props.get(f"EVI_{mm}"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN DOWNLOAD FUNCTION
# ---------------------------------------------------------------------------

def download_vegetation(
    states: List[str],
    start_year: int,
    end_year: int,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Download vegetation indices for all specified states and years.

    Writes:
      - monthly growing-season rows (raw/vegetation_monthly.csv)
      - growing-season county-year aggregates (processed/vegetation.csv)

    Returns:
      A DataFrame of the last batch's yearly aggregates (not the full dataset).
    """
    MONTHLY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    YEARLY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    outputs_exist = (MONTHLY_OUTPUT_FILE.exists(), YEARLY_OUTPUT_FILE.exists())
    if resume and any(outputs_exist) and not CHECKPOINT_FILE.exists():
        raise ValueError(
            "Existing vegetation files have no resume checkpoint. Use --aggregate-only "
            "to rebuild locally, or --no-resume in a separate working copy to download afresh."
        )
    if resume and CHECKPOINT_FILE.exists() and not all(outputs_exist):
        raise ValueError("Resume checkpoint exists but an output is missing; use --no-resume to rebuild.")

    state_fips_map: Dict[str, str] = {}
    for abbr in states:
        fips_list = fetch_state_fips([abbr])
        if fips_list:
            state_fips_map[abbr] = str(fips_list[0])

    checkpoint = load_checkpoint() if resume else {"completed": []}

    # A fresh run should not append to old output files.
    if not resume:
        CHECKPOINT_FILE.unlink(missing_ok=True)
        for f in (MONTHLY_OUTPUT_FILE, YEARLY_OUTPUT_FILE):
            if f.exists():
                logging.warning(f"Not resuming: removing existing {f} to avoid duplicate rows")
                f.unlink()

    crop_names = (
        [CURRENT_MASK_CROP]
        if CURRENT_MASK_CROP
        else (MASK_CROPS if CROP_SPECIFIC_MASK else [None])
    )
    total_tasks = len(states) * (end_year - start_year + 1) * len(crop_names)
    already_completed = 0
    # Count only the keys for this run.
    requested_keys = set()
    for y in range(start_year, end_year + 1):
        for abbr in states:
            fips = state_fips_map.get(abbr)
            if fips:
                for crop_name in crop_names:
                    requested_keys.add(_key(fips, y, crop_name))
    already_completed = sum(1 for k in checkpoint.get("completed", []) if k in requested_keys)
    remaining = total_tasks - already_completed

    logging.info("=" * 60)
    logging.info("VEGETATION DATA DOWNLOAD")
    logging.info("=" * 60)
    logging.info(f"States: {states}")
    logging.info(f"Years: {start_year}-{end_year}")
    if CURRENT_MASK_CROP:
        logging.info(f"Active crop mask: {CURRENT_MASK_CROP}")
    elif CROP_SPECIFIC_MASK:
        logging.info(f"Crop-specific masks: {MASK_CROPS}")
    logging.info(f"Total tasks: {total_tasks}")
    logging.info(f"Already completed (within request): {already_completed}")
    logging.info(f"Remaining: {remaining}")
    logging.info("=" * 60)

    processed_idx = 0
    last_yearly_df: Optional[pd.DataFrame] = None

    for year in range(start_year, end_year + 1):
        for state_abbr in states:
            state_fips = state_fips_map.get(state_abbr)
            if not state_fips:
                logging.warning(f"Could not find FIPS for {state_abbr}, skipping")
                continue

            if resume and is_completed(checkpoint, state_fips, year, CURRENT_MASK_CROP):
                continue

            processed_idx += 1
            start_month, end_month = get_growing_season(state_abbr)
            crop_msg = f" {CURRENT_MASK_CROP}" if CURRENT_MASK_CROP else ""
            logging.info(
                f"[{processed_idx}/{remaining if remaining > 0 else total_tasks}] "
                f"{state_abbr} {year}{crop_msg} (season: months {start_month}-{end_month-1})"
            )

            t0 = time.time()

            try:
                monthly_df = get_monthly_vegetation_stats_with_retry(
                    state_fips=state_fips,
                    state_abbr=state_abbr,
                    year=year,
                    crop_name=CURRENT_MASK_CROP,
                )

                if monthly_df.empty:
                    logging.warning(f"  No data returned for {state_abbr} {year}{crop_msg} (NOT marking completed).")
                    continue

                monthly_df["state_abbr"] = state_abbr
                append_csv(monthly_df, MONTHLY_OUTPUT_FILE)

                yearly_chunk = aggregate_to_yearly(monthly_df)
                yearly_chunk["state_abbr"] = state_abbr

                cols = [
                    "state_fips", "state_abbr", "county_fips", "county_name", "year", "crop", "mask_year",
                    "ndvi_mean_year", "ndvi_min_year", "ndvi_max_year",
                    "evi_mean_year", "evi_min_year", "evi_max_year",
                    "ndvi_std_year","evi_std_year","ndvi_sum_year","evi_sum_year", "n_months"
                ]
                yearly_chunk = yearly_chunk[[c for c in cols if c in yearly_chunk.columns]]
                append_csv(yearly_chunk, YEARLY_OUTPUT_FILE)

                last_yearly_df = yearly_chunk

                elapsed = time.time() - t0
                logging.info(f"  ✓ {len(monthly_df)} monthly rows; {len(yearly_chunk)} county-year rows in {elapsed:.1f}s")

                # Mark done after both files are written.
                mark_completed(checkpoint, state_fips, year, CURRENT_MASK_CROP)

            except CODE_ERROR_TYPES as e:
                logging.error(f"  FATAL (code error, not transient): {type(e).__name__}: {e}")
                raise
            except Exception as e:
                # Try this state, year, and crop again on the next run.
                logging.error(f"  Failed (will retry on re-run): {e}")
                continue

    completed_now = set(load_checkpoint().get("completed", [])) if CHECKPOINT_FILE.exists() else set()
    missing = sorted(requested_keys - completed_now)
    logging.info("=" * 60)
    logging.info(
        f"Completion audit: expected={len(requested_keys)} "
        f"completed={len(requested_keys & completed_now)} missing={len(missing)}"
    )
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        logging.warning(f"Missing checkpoint keys: {preview}{suffix}")
        raise RuntimeError(f"Vegetation download incomplete: {len(missing)} tasks remain.")

    if last_yearly_df is None:
        logging.warning("No data downloaded in this run.")
        return pd.DataFrame()

    logging.info("\n✓ Vegetation download complete!")
    logging.info(f"Monthly file:  {MONTHLY_OUTPUT_FILE}")
    logging.info(f"Yearly file:   {YEARLY_OUTPUT_FILE}")
    logging.info(f"Checkpoint:    {CHECKPOINT_FILE}")

    return last_yearly_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download NDVI/EVI data with crop masking and state-specific seasons"
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        help=f"State abbreviations (default: {DEFAULT_STATES})"
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=ANALYSIS_YEARS[0],
        help=f"Start year (default: {ANALYSIS_YEARS[0]}; CDL crop mask needs >=2008)"
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=ANALYSIS_YEARS[1],
        help=f"End year (default: {ANALYSIS_YEARS[1]})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore checkpoint"
    )

    parser.add_argument("--study", action="store_true",
                        help="Use config.STUDY_STATES (the modelling scope) instead of "
                             "DEFAULT_STATES. Avoids retyping the list.")
    parser.add_argument("--no-crop-mask", action="store_true",
                        help="Average MODIS over the whole county instead of "
                             "cropland only. Ablation use; requires --variant.")
    parser.add_argument("--scale", type=int, default=None,
                        help="Reduction scale in metres. Default is the asset's "
                             "native scale. Ablation use; requires --variant.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory holding raw/ and processed/ for this "
                             "variant's output. Relative paths resolve from the "
                             "repository root. Default is data/. "
                             "Requires --variant.")
    parser.add_argument("--product", default=None, choices=list(MODIS_PRODUCTS),
                        help="MODIS vegetation product. Default is MOD13A3 "
                             "(monthly, 1 km). MOD13Q1 is 16-day at 250 m. "
                             "Requires --variant.")
    parser.add_argument("--crop-specific-mask", action="store_true",
                        help="Build one vegetation row per analysis crop using "
                             "crop-specific CDL codes. Writes to a variant file; "
                             "defaults to --variant crop_specific.")
    parser.add_argument("--mask-crops", nargs="+", default=None,
                        help="Analysis crop names for --crop-specific-mask. "
                             f"Default: {CROP_SPECIFIC_MASK_CROPS}")
    parser.add_argument("--mask-year-offset", type=int, default=0,
                        help="Year offset for crop-specific CDL masks. Use -1 "
                             "for previous-year crop masks.")
    parser.add_argument("--variant", default=None,
                        help="Write to vegetation_<VARIANT>.csv and its own "
                             "checkpoint instead of the production files. "
                             "Required whenever --no-crop-mask or --scale is set, "
                             "so an ablation run cannot overwrite real data.")
    parser.add_argument("--log-file", default=None,
                        help="Also write progress logs to this file.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Rebuild yearly features from local monthly data.")

    args = parser.parse_args()

    global USE_CROP_MASK, REDUCE_SCALE, CROP_SPECIFIC_MASK, MASK_CROPS, CURRENT_MASK_CROP, MASK_YEAR_OFFSET
    global MODIS_PRODUCT
    global MONTHLY_OUTPUT_FILE, YEARLY_OUTPUT_FILE, CHECKPOINT_FILE
    if args.no_crop_mask and args.crop_specific_mask:
        parser.error("--no-crop-mask and --crop-specific-mask are mutually exclusive.")
    if args.mask_crops and not args.crop_specific_mask:
        parser.error("--mask-crops requires --crop-specific-mask.")
    if args.mask_year_offset not in (0, -1):
        parser.error("--mask-year-offset must be 0 or -1.")
    if args.start_year > args.end_year:
        parser.error("Start year must not exceed end year.")
    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive.")
    if not args.aggregate_only and not args.no_crop_mask and args.start_year + args.mask_year_offset < 2008:
        parser.error("The requested mask year predates nationwide CDL coverage (2008).")
    if args.crop_specific_mask:
        try:
            for crop in args.mask_crops or CROP_SPECIFIC_MASK_CROPS:
                get_cdl_codes(crop)
        except KeyError as exc:
            parser.error(str(exc))
    setup_file_logging(args.log_file)
    if args.crop_specific_mask and not args.variant:
        args.variant = "crop_specific_prev_year" if args.mask_year_offset == -1 else "crop_specific"
    if args.variant and (not args.variant.isascii() or not all(
        char.isalnum() or char in "_-" for char in args.variant
    )):
        parser.error("--variant may contain only ASCII letters, numbers, underscores, and hyphens.")
    if args.variant:
        # Naming and offset must agree, so previous-year data cannot land in a
        # same-year file.
        if ("prev_year" in args.variant) != (args.mask_year_offset == -1):
            parser.error("The variant name does not match --mask-year-offset.")
    if (args.no_crop_mask or args.scale or args.product or args.mask_year_offset
            or args.out_dir) and not args.variant:
        parser.error("--no-crop-mask, --scale, --product, --mask-year-offset and "
                     "--out-dir change where or what the data is, so they need "
                     "--variant to keep the production files intact.")
    if args.variant:
        v = args.variant
        # A variant belongs to whoever asked for it, so the caller says where
        # it goes. Without --out-dir it lands in data/.
        out_dir = Path(args.out_dir) if args.out_dir else DATA_PATH
        if not out_dir.is_absolute():
            out_dir = DATA_PATH.parent / out_dir
        raw_dir = out_dir / "raw"
        processed_dir = out_dir / "processed"
        MONTHLY_OUTPUT_FILE = raw_dir / f"vegetation_monthly_{v}.csv"
        YEARLY_OUTPUT_FILE = processed_dir / f"vegetation_{v}.csv"
        CHECKPOINT_FILE = raw_dir / f"vegetation_checkpoint_{v}.json"
    USE_CROP_MASK = not args.no_crop_mask
    REDUCE_SCALE = args.scale
    if args.product:
        MODIS_PRODUCT = MODIS_PRODUCTS[args.product]
    CROP_SPECIFIC_MASK = bool(args.crop_specific_mask)
    MASK_CROPS = list(args.mask_crops or CROP_SPECIFIC_MASK_CROPS)
    MASK_YEAR_OFFSET = int(args.mask_year_offset)
    logging.info(f"  crop mask: {'on' if USE_CROP_MASK else 'OFF'} | "
                 f"crop-specific: {'on' if CROP_SPECIFIC_MASK else 'off'} | "
                 f"mask-year offset: {MASK_YEAR_OFFSET} | "
                 f"scale: {REDUCE_SCALE or 'native'} | "
                 f"output: {YEARLY_OUTPUT_FILE.name}")
    states = resolve_states(args.states, args.study)
    resume = args.resume and not args.no_resume

    if args.aggregate_only:
        rebuild_from_file(MONTHLY_OUTPUT_FILE, YEARLY_OUTPUT_FILE)
        return

    authenticate_gee()

    if CROP_SPECIFIC_MASK:
        if not resume:
            for f in (MONTHLY_OUTPUT_FILE, YEARLY_OUTPUT_FILE, CHECKPOINT_FILE):
                if f.exists():
                    logging.warning(f"Not resuming: removing existing {f}")
                    f.unlink()
            resume = True

        # Download one crop at a time and append all crops to the same table.
        CROP_SPECIFIC_MASK = False
        for crop_name in MASK_CROPS:
            CURRENT_MASK_CROP = crop_name
            logging.info(f"Crop-specific vegetation pass: {crop_name}")
            download_vegetation(
                states=states,
                start_year=args.start_year,
                end_year=args.end_year,
                resume=resume,
            )
        CURRENT_MASK_CROP = None
        CROP_SPECIFIC_MASK = True
        return

    download_vegetation(
        states=states,
        start_year=args.start_year,
        end_year=args.end_year,
        resume=resume,
    )


if __name__ == "__main__":
    main()
