"""Compare crop masks at June-September yield-prediction cutoffs.

Run: python src/compare_masks_by_month.py
Reads local monthly inputs and writes comparison tables under results/.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from config import get_growing_season

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CLIMATE_FEATURES = [
    "TMIN", "TMAX", "PRCP", "VPD", "ETO", "SRAD",
    "GDD_TMAX", "EDD_TMAX", "HOT_DAY_FRAC", "TMAX_MAX",
]

SOIL_FEATURES = ["clay_mean", "ph_mean", "soc_mean", "bdod_mean"]

VEG_FEATURES = [
    "evi_min_year", "evi_mean_year", "evi_max_year",
    "ndvi_min_year", "ndvi_mean_year", "ndvi_max_year",
]

MEAN_CLIMATE = ["TMIN", "TMAX", "VPD", "SRAD", "WIND"]

RATE_CLIMATE = ["PRCP", "ETO", "GDD_TMAX", "EDD_TMAX"]

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "n_jobs": -1,
}

SEEDS = (25, 26, 27, 28, 29)   # random forest runs are averaged over these seeds
BOOTSTRAP_DRAWS = 2000         # county resamples for the 95% intervals
BOOTSTRAP_SEED = 12345         # fixed so the intervals repeat exactly
SPATIAL_FOLDS = 5              # county-held-out folds
CROPS = ("CORN", "SOYBEANS")
CUTOFFS = (6, 7, 8, 9)         # months models are fitted at (June to September)

TABLE_CUTOFFS = (4, 5, 6, 7, 8, 9) # months saved in the feature table (April to September)

MODELS = ("random_forest", "hist_gradient_boosting")
BASE_ARMS = (
    "cropland",
    "same_year_oracle",
    "previous_year",
)

# Arms whose mask has no crop dimension, so they join on county and year only.
GENERIC_ARMS = ("cropland", "previous_year_cropland")
ARMS = BASE_ARMS

MERGED = DATA / "processed" / "merged.csv"
CLIMATE_MONTHLY = DATA / "raw" / "climate_monthly.csv"

def veg_monthly(suffix: str = "") -> dict[str, Path]:
    """Monthly vegetation files for a MODIS product.

    An empty suffix is the 1 km MOD13A3 set. "_250m" is the MOD13Q1 set built
    by download_vegetation.py --product MOD13Q1.
    """
    return {
        "cropland": DATA / "raw" / f"vegetation_monthly{suffix}.csv",
        "same_year_oracle": DATA / "raw" / f"vegetation_monthly_crop_specific{suffix}.csv",
        "previous_year": DATA / "raw" / f"vegetation_monthly_crop_specific_prev_year{suffix}.csv",
        "previous_year_cropland": DATA / "raw" / f"vegetation_monthly_prev_year_cropland{suffix}.csv",
    }


PRODUCT_SUFFIX = {"MOD13A3": "", "MOD13Q1": "_250m"}
VEG_MONTHLY = veg_monthly()
OUT_DIR = ROOT / "results"


def select_product(name: str) -> str:
    """Point this module at one product's files. Returns the name suffix.

    The scripts that import this module call it before audit_inputs(), so
    they all read the same product.
    """
    global VEG_MONTHLY, FEATURE_TABLE
    suffix = PRODUCT_SUFFIX[name]
    VEG_MONTHLY = veg_monthly(suffix)
    FEATURE_TABLE = DATA / "processed" / f"features_by_cutoff{suffix}.csv"
    # Only the three main arms are required. previous_year_cropland is
    # optional and checked when its flag asks for it.
    missing = [str(VEG_MONTHLY[arm]) for arm in BASE_ARMS if not VEG_MONTHLY[arm].exists()]
    if missing:
        raise SystemExit(f"Missing vegetation files for {name}: {missing}")
    return suffix
FEATURE_TABLE = DATA / "processed" / "features_by_cutoff.csv"

log = logging.getLogger(__name__)


def setup_logging(log_file: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not log_file:
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"county_fips": str, "state_fips": str})
    if "county_fips" in df:
        df["county_fips"] = df["county_fips"].str.zfill(5)
    return df


def load_base_table() -> pd.DataFrame:
    """Load yield and soil values for the study's crop-county-year sample."""
    base = read_csv(MERGED)
    columns = ["crop", "state_abbr", "county_fips", "year", "yield_value"] + SOIL_FEATURES
    return base.loc[base["crop"].isin(CROPS), columns].drop_duplicates(
        ["crop", "county_fips", "year"]
    )


def audit_crop_mask_file(path: Path, expected_offset: int, arm: str) -> tuple[pd.DataFrame, list[dict]]:
    df = read_csv(path)
    keys = ["crop", "county_fips", "year", "month"]
    required = keys + ["mask_year", "ndvi_mean", "evi_mean", "state_abbr"]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    duplicate_rows = int(df.duplicated(keys).sum())
    bad_mask_year = int((df["mask_year"] != df["year"] + expected_offset).sum())
    missing_values = int(df[["ndvi_mean", "evi_mean"]].isna().any(axis=1).sum())
    months_per_key = df.groupby(["crop", "county_fips", "year"]).size()

    checks = [
        {"arm": arm, "check": "rows", "value": len(df), "passed": True, "required": False},
        {"arm": arm, "check": "duplicate_crop_county_year_month", "value": duplicate_rows,
         "passed": duplicate_rows == 0, "required": True},
        {"arm": arm, "check": "wrong_mask_year", "value": bad_mask_year,
         "passed": bad_mask_year == 0, "required": True},
        {"arm": arm, "check": "missing_ndvi_or_evi", "value": missing_values,
         "passed": missing_values == 0, "required": False},
        {"arm": arm, "check": "minimum_months_per_crop_county_year",
         "value": int(months_per_key.min()), "passed": int(months_per_key.min()) == 9,
         "required": True},
        {"arm": arm, "check": "maximum_months_per_crop_county_year",
         "value": int(months_per_key.max()), "passed": int(months_per_key.max()) == 9,
         "required": True},
    ]
    return df, checks


def audit_inputs():
    current, current_checks = audit_crop_mask_file(
        VEG_MONTHLY["same_year_oracle"], 0, "same_year_oracle"
    )
    previous, previous_checks = audit_crop_mask_file(
        VEG_MONTHLY["previous_year"], -1, "previous_year"
    )

    keys = ["crop", "county_fips", "year", "month"]
    current_keys = pd.MultiIndex.from_frame(current[keys].sort_values(keys))
    previous_keys = pd.MultiIndex.from_frame(previous[keys].sort_values(keys))
    same_keys = current_keys.equals(previous_keys)
    checks = current_checks + previous_checks + [{
        "arm": "current_vs_previous",
        "check": "same_keys_in_same_order",
        "value": int(same_keys),
        "passed": same_keys,
        "required": True,
    }]

    audit = pd.DataFrame(checks)
    failed = audit.loc[audit["required"] & ~audit["passed"]]
    if not failed.empty:
        raise ValueError(f"Crop-mask input audit failed:\n{failed.to_string(index=False)}")

    frames = {
        "cropland": read_csv(VEG_MONTHLY["cropland"]),
        "same_year_oracle": current,
        "previous_year": previous,
    }
    other_crop = previous.copy()
    other_crop["crop"] = other_crop["crop"].map({
        "CORN": "SOYBEANS",
        "SOYBEANS": "CORN",
    })
    frames["previous_year_other_crop"] = other_crop
    path = VEG_MONTHLY.get("previous_year_cropland")
    if path is not None and path.exists():
        frames["previous_year_cropland"] = read_csv(path)
    return frames, audit


def add_cutoff_bounds(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    out = df.copy()
    bounds = out["state_abbr"].map(get_growing_season)
    out["_start"] = [item[0] for item in bounds]
    out["_end"] = [min(item[1] - 1, cutoff) for item in bounds]
    return out[(out["month"] >= out["_start"]) & (out["month"] <= out["_end"])]


def aggregate_climate(monthly: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    df = add_cutoff_bounds(monthly, cutoff)
    keys = ["state_abbr", "county_fips", "year"]

    for column in MEAN_CLIMATE:
        df[f"_weighted_{column}"] = df[column] * df["N_DAYS"]

    named = {
        "observed_days": ("N_DAYS", "sum"),
        "climate_months": ("month", "count"),
        "HOT_DAYS": ("HOT_DAYS", "sum"),
        "TMAX_MAX": ("TMAX_MAX", "max"),
    }
    for column in MEAN_CLIMATE:
        named[f"_weighted_{column}"] = (f"_weighted_{column}", "sum")
    for column in RATE_CLIMATE:
        named[column] = (column, "sum")

    out = df.groupby(keys, as_index=False).agg(**named)
    for column in MEAN_CLIMATE:
        out[column] = out[f"_weighted_{column}"] / out["observed_days"]
        out = out.drop(columns=f"_weighted_{column}")
    for column in RATE_CLIMATE:
        out[column] = out[column] / out["observed_days"]
    out["HOT_DAY_FRAC"] = out["HOT_DAYS"] / out["observed_days"]
    return out


def aggregate_vegetation(monthly: pd.DataFrame, cutoff: int, arm: str) -> pd.DataFrame:
    df = add_cutoff_bounds(monthly, cutoff)
    keys = ["county_fips", "year"]
    if "crop" in df.columns:
        keys.insert(0, "crop")

    out = df.groupby(keys, as_index=False).agg(
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_max_year=("ndvi_mean", "max"),
        evi_min_year=("evi_mean", "min"),
        evi_mean_year=("evi_mean", "mean"),
        evi_max_year=("evi_mean", "max"),
        vegetation_months=("month", "count"),
    )
    renamed = {column: f"{column}__{arm}" for column in VEG_FEATURES}
    renamed["vegetation_months"] = f"vegetation_months__{arm}"
    return out.rename(columns=renamed)


def expected_months(state: str, cutoff: int) -> int:
    start, end_exclusive = get_growing_season(state)
    return max(0, min(end_exclusive - 1, cutoff) - start + 1)


def build_cutoff_table(
    base: pd.DataFrame,
    climate: pd.DataFrame,
    vegetation: dict[str, pd.DataFrame],
    cutoff: int,
) -> pd.DataFrame:
    climate_cutoff = aggregate_climate(climate, cutoff)
    df = base.merge(climate_cutoff, on=["state_abbr", "county_fips", "year"], how="inner")

    for arm, monthly in vegetation.items():
        veg = aggregate_vegetation(monthly, cutoff, arm)
        keys = (["county_fips", "year"] if arm in GENERIC_ARMS
                else ["crop", "county_fips", "year"])
        df = df.merge(veg, on=keys, how="inner")

    needed = CLIMATE_FEATURES + SOIL_FEATURES
    for arm in vegetation:
        needed.extend(f"{column}__{arm}" for column in VEG_FEATURES)
    df = df.dropna(subset=needed + ["yield_value"])

    expected = df["state_abbr"].map(lambda state: expected_months(state, cutoff))
    complete = df["climate_months"].eq(expected)
    for arm in vegetation:
        complete &= df[f"vegetation_months__{arm}"].eq(expected)
    return df.loc[complete].copy()


def feature_columns(arm: str) -> list[str]:
    vegetation = [f"{column}__{arm}" for column in VEG_FEATURES]
    return CLIMATE_FEATURES + vegetation + SOIL_FEATURES


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    years = sorted(df["year"].unique())
    test_years = years[-4:]
    return df[~df["year"].isin(test_years)], df[df["year"].isin(test_years)], test_years


def make_model(model_name: str, seed: int):
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed, **RF_PARAMS)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    seeds: tuple[int, ...],
    model_name: str):

    model_seeds = seeds if model_name == "random_forest" else seeds[:1]
    result = np.zeros(len(test))

    for seed in model_seeds:
        model = make_model(model_name, seed)
        model.fit(train[columns], train["yield_value"])
        result += model.predict(test[columns])
    return result / len(model_seeds)


def county_mean(train: pd.DataFrame, test: pd.DataFrame):
    values = train.groupby("county_fips")["yield_value"].mean()
    return test["county_fips"].map(values).fillna(train["yield_value"].mean()).to_numpy()


def county_trend(train: pd.DataFrame, test: pd.DataFrame):
    means = train.groupby("county_fips")["yield_value"].mean()
    years = train.groupby("county_fips")["year"].mean()
    yield_deviation = train["yield_value"] - train["county_fips"].map(means)
    year_deviation = train["year"] - train["county_fips"].map(years)
    denominator = float(np.dot(year_deviation, year_deviation))
    slope = float(np.dot(year_deviation, yield_deviation) / denominator)
    base = test["county_fips"].map(means).fillna(train["yield_value"].mean()).to_numpy()
    test_years = test["year"] - test["county_fips"].map(years).fillna(train["year"].mean())
    return base + slope * test_years.to_numpy()


def skill(squared_error: np.ndarray, actual: np.ndarray, baseline: np.ndarray):
    baseline_error = float(np.sum((actual - baseline) ** 2))
    return 1.0 - float(squared_error.sum()) / baseline_error


def bootstrap_q(
    arm_error: np.ndarray,
    reference_error: np.ndarray,
    counties: np.ndarray,
    draws: int):

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique = np.unique(counties)
    positions = {county: np.where(counties == county)[0] for county in unique}
    values = []
    for _ in range(draws):
        sample = rng.choice(unique, len(unique), replace=True)
        rows = np.concatenate([positions[county] for county in sample])
        values.append(np.sqrt(arm_error[rows].mean() / reference_error[rows].mean()))
    return tuple(np.percentile(values, [2.5, 97.5]))


def result_rows(
    crop: str,
    cutoff: int,
    model_name: str,
    evaluation: str,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    counties: np.ndarray,
    bootstrap_draws: int,
    n_train: int | float,
    n_folds: int,
    test_years: str,
    mean_baseline: np.ndarray | None = None,
    trend_baseline: np.ndarray | None = None):

    errors = {arm: (actual - values) ** 2 for arm, values in predictions.items()}
    rows: list[dict] = []
    for arm in ARMS:
        rmse = float(np.sqrt(errors[arm].mean()))
        q_crop = rmse / float(np.sqrt(errors["cropland"].mean()))
        q_oracle = rmse / float(np.sqrt(errors["same_year_oracle"].mean()))

        crop_ci = (np.nan, np.nan)
        oracle_ci = (np.nan, np.nan)
        if arm != "cropland":
            crop_ci = bootstrap_q(errors[arm], errors["cropland"], counties, bootstrap_draws)
        if arm != "same_year_oracle":
            oracle_ci = bootstrap_q(
                errors[arm], errors["same_year_oracle"], counties, bootstrap_draws
            )

        rows.append({
            "crop": crop,
            "cutoff_month": cutoff,
            "model": model_name,
            "evaluation": evaluation,
            "arm": arm,
            "n_train": n_train,
            "n_test": len(actual),
            "n_folds": n_folds,
            "test_years": test_years,
            "r2": r2_score(actual, predictions[arm]),
            "rmse": rmse,
            "q_vs_cropland": q_crop,
            "q_vs_cropland_lo": crop_ci[0],
            "q_vs_cropland_hi": crop_ci[1],
            "q_vs_same_year_oracle": q_oracle,
            "q_vs_same_year_oracle_lo": oracle_ci[0],
            "q_vs_same_year_oracle_hi": oracle_ci[1],
            "skill_vs_county_mean": (
                skill(errors[arm], actual, mean_baseline)
                if mean_baseline is not None else np.nan
            ),
            "skill_vs_county_trend": (
                skill(errors[arm], actual, trend_baseline)
                if trend_baseline is not None else np.nan
            ),
        })
    return rows


def evaluate_temporal(
    crop: str,
    cutoff: int,
    df: pd.DataFrame,
    seeds: tuple,
    bootstrap_draws: int,
    model_names: tuple):

    crop_df = df[df["crop"] == crop].copy()
    train, test, test_years = temporal_split(crop_df)
    actual = test["yield_value"].to_numpy()
    counties = test["county_fips"].to_numpy()
    rows = []
    by_year = []

    for model_name in model_names:
        predictions = {}
        for arm in ARMS:
            log.info(f"  temporal {crop} cutoff={cutoff} model={model_name} arm={arm}")
            predictions[arm] = predict(
                train, test, feature_columns(arm), seeds, model_name
            )

        rows.extend(result_rows(
            crop=crop,
            cutoff=cutoff,
            model_name=model_name,
            evaluation="temporal_holdout",
            actual=actual,
            predictions=predictions,
            counties=counties,
            bootstrap_draws=bootstrap_draws,
            n_train=len(train),
            n_folds=1,
            test_years=f"{test_years[0]}-{test_years[-1]}",
            mean_baseline=county_mean(train, test),
            trend_baseline=county_trend(train, test),
        ))

        errors = {arm: (actual - values) ** 2 for arm, values in predictions.items()}
        for arm in ARMS:
            for year in test_years:
                positions = test["year"].to_numpy() == year
                by_year.append({
                    "crop": crop,
                    "cutoff_month": cutoff,
                    "model": model_name,
                    "year": year,
                    "arm": arm,
                    "n": int(positions.sum()),
                    "rmse": float(np.sqrt(errors[arm][positions].mean())),
                    "q_vs_cropland": float(np.sqrt(
                        errors[arm][positions].mean()
                        / errors["cropland"][positions].mean()
                    )),
                })
    return rows, by_year


def evaluate_spatial(
    crop: str,
    cutoff: int,
    df: pd.DataFrame,
    seeds: tuple[int, ...],
    bootstrap_draws: int,
    model_names: tuple[str, ...]):

    crop_df = df[df["crop"] == crop].copy()
    groups = crop_df["county_fips"]
    splitter = GroupKFold(n_splits=SPATIAL_FOLDS)
    folds = list(splitter.split(crop_df, crop_df["yield_value"], groups=groups))
    rows = []

    for model_name in model_names:
        actual_parts = []
        county_parts = []
        prediction_parts = {arm: [] for arm in ARMS}
        for fold_number, (train_index, test_index) in enumerate(folds, start=1):
            train = crop_df.iloc[train_index]
            test = crop_df.iloc[test_index]
            actual_parts.append(test["yield_value"].to_numpy())
            county_parts.append(test["county_fips"].to_numpy())
            for arm in ARMS:
                log.info(
                    f"  spatial {crop} cutoff={cutoff} model={model_name} "
                    f"fold={fold_number}/{SPATIAL_FOLDS} arm={arm}"
                )
                prediction_parts[arm].append(
                    predict(train, test, feature_columns(arm), seeds, model_name)
                )

        actual = np.concatenate(actual_parts)
        counties = np.concatenate(county_parts)
        predictions = {
            arm: np.concatenate(parts) for arm, parts in prediction_parts.items()
        }
        rows.extend(result_rows(
            crop=crop,
            cutoff=cutoff,
            model_name=model_name,
            evaluation="county_heldout_cv",
            actual=actual,
            predictions=predictions,
            counties=counties,
            bootstrap_draws=bootstrap_draws,
            n_train=np.nan,
            n_folds=SPATIAL_FOLDS,
            test_years=f"{int(crop_df['year'].min())}-{int(crop_df['year'].max())}",
        ))
    return rows


def main():
    global ARMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--n-estimators", type=int, default=RF_PARAMS["n_estimators"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--skip-spatial", action="store_true")
    parser.add_argument("--include-rotation-proxy", action="store_true")
    parser.add_argument("--include-prev-year-cropland", action="store_true",
                        help="Add a previous-year generic cropland arm, so a stale "
                             "crop map can be told apart from a stale land-cover map.")
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument("--product", choices=sorted(PRODUCT_SUFFIX), default="MOD13Q1",
                        help="MODIS product to read. MOD13Q1 is the 250 m set; "
                             "its results are written with a _250m suffix.")
    parser.add_argument("--match-product", choices=sorted(PRODUCT_SUFFIX), default=None,
                        help="Keep only county-years this product also has, so two "
                             "products can be compared on the same sample. Results "
                             "get a _matched suffix.")
    args = parser.parse_args()
    if args.table_only and args.include_rotation_proxy:
        parser.error("--table-only only writes the main three-arm table")
    setup_logging(args.log_file)
    suffix = select_product(args.product)

    # Restricting to the other product's county-years makes the two runs
    # comparable; the products drop different rows for missing data.
    match_keys = None
    if args.match_product:
        if args.match_product == args.product:
            parser.error("--match-product must differ from --product.")
        other = FEATURE_TABLE.with_name(
            f"features_by_cutoff{PRODUCT_SUFFIX[args.match_product]}.csv")
        if not other.exists():
            raise SystemExit(f"Need {other} first; build it with --table-only.")
        keys = pd.read_csv(other, usecols=["cutoff_month", "crop", "county_fips", "year"],
                           dtype={"county_fips": str})
        match_keys = {
            cutoff: pd.MultiIndex.from_frame(part[["crop", "county_fips", "year"]])
            for cutoff, part in keys.groupby("cutoff_month")
        }
        suffix = f"{suffix}_matched"

    RF_PARAMS["n_estimators"] = args.n_estimators
    seeds = tuple(args.seeds)
    if args.include_rotation_proxy and args.include_prev_year_cropland:
        parser.error("Run the rotation proxy and the previous-year cropland arm separately.")
    if args.include_rotation_proxy:
        ARMS = BASE_ARMS + ("previous_year_other_crop",)
    if args.include_prev_year_cropland:
        needed = VEG_MONTHLY["previous_year_cropland"]
        if not needed.exists():
            raise SystemExit(f"Missing {needed}")
        ARMS = BASE_ARMS + ("previous_year_cropland",)

    vegetation, audit = audit_inputs()
    vegetation = {arm: vegetation[arm] for arm in ARMS}
    climate = read_csv(CLIMATE_MONTHLY)
    base = load_base_table()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = OUT_DIR / f"input_checks{suffix}.csv"
    audit.to_csv(audit_path, index=False)
    log.info("Crop-mask input audit passed.")

    rows = []
    by_year = []
    tables = []
    for cutoff in TABLE_CUTOFFS:
        table = build_cutoff_table(base, climate, vegetation, cutoff)
        if match_keys is not None:
            before = len(table)
            table = table[
                pd.MultiIndex.from_frame(table[["crop", "county_fips", "year"]])
                .isin(match_keys[cutoff])
            ].copy()
            log.info(f"Cutoff month {cutoff}: {len(table):,} matched rows "
                     f"({before - len(table):,} dropped to match {args.match_product})")
        else:
            log.info(f"Cutoff month {cutoff}: {len(table):,} matched rows")
        saved = table.copy()
        saved.insert(0, "cutoff_month", cutoff)
        tables.append(saved)
        if args.table_only or cutoff not in CUTOFFS:
            continue
        for crop in CROPS:
            crop_rows, year_rows = evaluate_temporal(
                crop, cutoff, table, seeds, args.bootstrap, tuple(args.models)
            )
            rows.extend(crop_rows)
            by_year.extend(year_rows)
            if not args.skip_spatial:
                rows.extend(evaluate_spatial(
                    crop, cutoff, table, seeds, args.bootstrap, tuple(args.models)
                ))

    # Only the main three-arm run writes the table, so it matches the
    # reported models. The rotation-proxy and matched runs leave it alone.
    if (not args.include_rotation_proxy and not args.include_prev_year_cropland
            and match_keys is None):
        FEATURE_TABLE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_csv(FEATURE_TABLE, index=False)
        log.info(f"Wrote {FEATURE_TABLE}")
    if args.table_only:
        return

    results = pd.DataFrame(rows)
    yearly = pd.DataFrame(by_year)
    if args.include_rotation_proxy:
        stem = "masks_rotation"
    elif args.include_prev_year_cropland:
        stem = "masks_prev_year_cropland"
    else:
        stem = "masks_by_month"
    results_path = OUT_DIR / f"{stem}{suffix}.csv"
    yearly_path = OUT_DIR / f"{stem}{suffix}_by_year.csv"
    results.to_csv(results_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    log.info(f"Wrote {results_path}")
    log.info(f"Wrote {yearly_path}")
    log.info(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
