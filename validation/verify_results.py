"""Check that a separate calculation reproduces the August and September corn results.

Start from the monthly CSVs, rebuild the inputs, and fit both models for the
three main masks. Compare RMSE and the ratio to generic cropland with the
saved results. This covers both the year holdout and county-held-out tests.
It uses the same growing-season settings but keeps its own calculation code.

Run: python validation/verify_results.py --product MOD13Q1
Writes results/verify_results<suffix>.csv.
"""

import argparse
import logging
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import get_growing_season


RESULTS = ROOT / "results"
MONTHS = (8, 9)
ARMS = ("cropland", "same_year_oracle", "previous_year")
MODELS = ("random_forest", "hist_gradient_boosting")
SEEDS = (25, 26, 27, 28, 29)  # Must match the model seeds in compare_masks_by_month.py.

CLIMATE_COLUMNS = [
    "TMIN", "TMAX", "PRCP", "VPD", "ETO", "SRAD",
    "GDD_TMAX", "EDD_TMAX", "HOT_DAY_FRAC", "TMAX_MAX",
]
SOIL_COLUMNS = ["clay_mean", "ph_mean", "soc_mean", "bdod_mean"]
VEG_COLUMNS = [
    "evi_min_year", "evi_mean_year", "evi_max_year",
    "ndvi_min_year", "ndvi_mean_year", "ndvi_max_year",
]
# Set by --product. This file deliberately does not import the primary
# analysis program, so the paths are repeated here rather than shared.
SUFFIX = ""
VEG_FILES = {}


def set_product(name: str) -> None:
    global SUFFIX, VEG_FILES
    SUFFIX = {"MOD13A3": "", "MOD13Q1": "_250m"}[name]
    VEG_FILES = {
        "cropland": ROOT / "data" / "raw" / f"vegetation_monthly{SUFFIX}.csv",
        "same_year_oracle": ROOT / "data" / "raw" / f"vegetation_monthly_crop_specific{SUFFIX}.csv",
        "previous_year": ROOT / "data" / "raw" / f"vegetation_monthly_crop_specific_prev_year{SUFFIX}.csv",
    }


set_product("MOD13A3")

log = logging.getLogger(__name__)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"county_fips": str, "state_fips": str})
    if "county_fips" in df:
        df["county_fips"] = df["county_fips"].str.zfill(5)
    return df


def keep_season_months(df: pd.DataFrame, last_month: int) -> pd.DataFrame:
    bounds = df["state_abbr"].map(get_growing_season)
    start = np.array([value[0] for value in bounds])
    end = np.array([min(value[1] - 1, last_month) for value in bounds])
    return df[(df["month"] >= start) & (df["month"] <= end)].copy()


def climate_table(monthly: pd.DataFrame, last_month: int) -> pd.DataFrame:
    df = keep_season_months(monthly, last_month)
    keys = ["state_abbr", "county_fips", "year"]
    means = ["TMIN", "TMAX", "VPD", "SRAD", "WIND"]
    rates = ["PRCP", "ETO", "GDD_TMAX", "EDD_TMAX"]
    for name in means:
        df[f"weighted_{name}"] = df[name] * df["N_DAYS"]

    rules = {
        "days": ("N_DAYS", "sum"),
        "month_count": ("month", "count"),
        "hot_days": ("HOT_DAYS", "sum"),
        "TMAX_MAX": ("TMAX_MAX", "max"),
    }
    for name in means:
        rules[f"weighted_{name}"] = (f"weighted_{name}", "sum")
    for name in rates:
        rules[name] = (name, "sum")

    result = df.groupby(keys, as_index=False).agg(**rules)
    for name in means:
        result[name] = result[f"weighted_{name}"] / result["days"]
    for name in rates:
        result[name] = result[name] / result["days"]
    result["HOT_DAY_FRAC"] = result["hot_days"] / result["days"]
    return result


def vegetation_table(monthly: pd.DataFrame, last_month: int, arm: str) -> pd.DataFrame:
    df = keep_season_months(monthly, last_month)
    keys = ["county_fips", "year"]
    if "crop" in df:
        keys.insert(0, "crop")
    result = df.groupby(keys, as_index=False).agg(
        evi_min_year=("evi_mean", "min"),
        evi_mean_year=("evi_mean", "mean"),
        evi_max_year=("evi_mean", "max"),
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_max_year=("ndvi_mean", "max"),
        veg_month_count=("month", "count"),
    )
    rename = {name: f"{name}__{arm}" for name in VEG_COLUMNS}
    rename["veg_month_count"] = f"veg_month_count__{arm}"
    return result.rename(columns=rename)


def build_table(last_month: int) -> pd.DataFrame:
    merged = read_csv(ROOT / "data" / "processed" / "merged.csv")
    base = merged[merged["crop"] == "CORN"][
        ["crop", "state_abbr", "county_fips", "year", "yield_value"] + SOIL_COLUMNS
    ].drop_duplicates(["crop", "county_fips", "year"])

    climate = climate_table(read_csv(ROOT / "data" / "raw" / "climate_monthly.csv"), last_month)
    result = base.merge(climate, on=["state_abbr", "county_fips", "year"])
    for arm, path in VEG_FILES.items():
        vegetation = vegetation_table(read_csv(path), last_month, arm)
        keys = ["county_fips", "year"] if arm == "cropland" else ["crop", "county_fips", "year"]
        result = result.merge(vegetation, on=keys)

    needed = CLIMATE_COLUMNS + SOIL_COLUMNS
    for arm in ARMS:
        needed += [f"{name}__{arm}" for name in VEG_COLUMNS]
    result = result.dropna(subset=needed + ["yield_value"])

    expected = result["state_abbr"].map(
        lambda state: min(get_growing_season(state)[1] - 1, last_month)
        - get_growing_season(state)[0] + 1
    )
    complete = result["month_count"].eq(expected)
    for arm in ARMS:
        complete &= result[f"veg_month_count__{arm}"].eq(expected)
    return result[complete].reset_index(drop=True)


def feature_names(arm: str) -> list[str]:
    return CLIMATE_COLUMNS + [f"{name}__{arm}" for name in VEG_COLUMNS] + SOIL_COLUMNS


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, arm: str, model_name: str) -> np.ndarray:
    columns = feature_names(arm)
    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=SEEDS[0],
        )
        model.fit(train[columns], train["yield_value"])
        return model.predict(test[columns])

    predictions = np.zeros(len(test))
    for seed in SEEDS:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            max_features=0.5, n_jobs=-1, random_state=seed,
        )
        model.fit(train[columns], train["yield_value"])
        predictions += model.predict(test[columns])
    return predictions / len(SEEDS)


def score(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def run_temporal(table: pd.DataFrame, month: int, model_name: str) -> list[dict]:
    test_years = sorted(table["year"].unique())[-4:]
    train = table[~table["year"].isin(test_years)]
    test = table[table["year"].isin(test_years)]
    return score_arms(train, test, month, model_name, "temporal_holdout")


def run_spatial(table: pd.DataFrame, month: int, model_name: str) -> list[dict]:
    pieces = {arm: [] for arm in ARMS}
    actual_parts = []
    splitter = GroupKFold(n_splits=5)
    for train_index, test_index in splitter.split(table, groups=table["county_fips"]):
        train = table.iloc[train_index]
        test = table.iloc[test_index]
        actual_parts.append(test["yield_value"].to_numpy())
        for arm in ARMS:
            pieces[arm].append(fit_predict(train, test, arm, model_name))

    actual = np.concatenate(actual_parts)
    rmses = {arm: score(actual, np.concatenate(pieces[arm])) for arm in ARMS}
    return make_rows(month, model_name, "county_heldout_cv", rmses)


def score_arms(
    train: pd.DataFrame,
    test: pd.DataFrame,
    month: int,
    model_name: str,
    evaluation: str,
) -> list[dict]:
    actual = test["yield_value"].to_numpy()
    rmses = {
        arm: score(actual, fit_predict(train, test, arm, model_name))
        for arm in ARMS
    }
    return make_rows(month, model_name, evaluation, rmses)


def make_rows(month: int, model_name: str, evaluation: str, rmses: dict[str, float]) -> list[dict]:
    return [{
        "crop": "CORN",
        "cutoff_month": month,
        "model": model_name,
        "evaluation": evaluation,
        "arm": arm,
        "verification_rmse": rmses[arm],
        "verification_q_vs_cropland": rmses[arm] / rmses["cropland"],
    } for arm in ARMS]


def compare_results(result: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    """Require every expected comparison once before comparing scores."""
    keys = ["crop", "cutoff_month", "model", "evaluation", "arm"]
    expected = set(product(
        ("CORN",), MONTHS, MODELS,
        ("temporal_holdout", "county_heldout_cv"), ARMS,
    ))
    primary = primary[primary["crop"].eq("CORN") & primary["cutoff_month"].isin(MONTHS)]
    for label, frame in (("Independent", result), ("Primary", primary)):
        actual = set(frame[keys].itertuples(index=False, name=None))
        if frame.duplicated(keys).any() or actual != expected:
            raise ValueError(f"{label} results have missing, unexpected, or duplicate comparisons.")
    primary = primary[keys + ["rmse", "q_vs_cropland"]].rename(columns={
        "rmse": "primary_rmse", "q_vs_cropland": "primary_q_vs_cropland",
    })
    result = result.merge(primary, on=keys, validate="one_to_one")
    result["rmse_abs_difference"] = (result["verification_rmse"] - result["primary_rmse"]).abs()
    result["q_abs_difference"] = (
        result["verification_q_vs_cropland"] - result["primary_q_vs_cropland"]
    ).abs()
    result["passed"] = (result["rmse_abs_difference"] < 1e-9) & (result["q_abs_difference"] < 1e-9)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["MOD13A3", "MOD13Q1"], default="MOD13Q1")
    set_product(parser.parse_args().product)
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(
            log_dir / "verify_results.log", encoding="utf-8"
        )],
    )
    rows = []
    for month in MONTHS:
        log.info("Building independent month %s table", month)
        table = build_table(month)
        for model_name in MODELS:
            log.info("Checking month %s with %s", month, model_name)
            rows += run_temporal(table, month, model_name)
            rows += run_spatial(table, month, model_name)

    result = pd.DataFrame(rows)
    primary = read_csv(RESULTS / f"masks_by_month{SUFFIX}.csv")
    result = compare_results(result, primary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"verify_results{SUFFIX}.csv"
    result.to_csv(output, index=False)
    log.info("Wrote %s", output)
    if not result["passed"].all():
        raise SystemExit("Independent verification did not match the primary results.")
    log.info("All %s independent comparisons passed", len(result))


if __name__ == "__main__":
    main()
