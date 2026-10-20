"""Compare crop-mask results separately for each state.

Run: python src/compare_masks_by_state.py
Writes results/masks_by_state<suffix>.csv.
"""

import argparse
import logging

import numpy as np
import pandas as pd

import compare_masks_by_month as masks

log = logging.getLogger(__name__)


def main() -> None:
    """Score each mask against generic cropland, state by state, in the future-year test."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-estimators", type=int, default=masks.RF_PARAMS["n_estimators"],
                        help="Trees per random forest (default: %(default)s).")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(masks.SEEDS),
                        help="Random forest seeds; predictions are averaged over them "
                             "(default: %(default)s).")
    parser.add_argument("--bootstrap", type=int, default=masks.BOOTSTRAP_DRAWS,
                        help="County resamples for the 95%% intervals (default: %(default)s).")
    parser.add_argument("--log-file", default=str(masks.ROOT / "logs" / "masks_by_state.log"),
                        help="Also write progress logs to this file (default: logs/masks_by_state.log).")
    parser.add_argument("--product", choices=sorted(masks.PRODUCT_SUFFIX), default="MOD13Q1",
                        help="MODIS product: MOD13A3 (1 km) or MOD13Q1 (250 m) "
                             "(default: %(default)s).")
    args = parser.parse_args()
    masks.setup_logging(args.log_file)
    suffix = masks.select_product(args.product)
    masks.RF_PARAMS["n_estimators"] = args.n_estimators
    seeds = tuple(args.seeds)

    vegetation, _ = masks.audit_inputs()
    vegetation = {arm: vegetation[arm] for arm in masks.BASE_ARMS}
    climate = masks.read_csv(masks.CLIMATE_MONTHLY)
    base = masks.load_base_table()

    rows = []
    for month in masks.CUTOFFS:
        table = masks.build_cutoff_table(base, climate, vegetation, month)
        for crop in masks.CROPS:
            crop_data = table[table["crop"] == crop].copy()
            train, test, test_years = masks.temporal_split(crop_data)
            actual = test["yield_value"].to_numpy()

            for model_name in masks.MODELS:
                # Squared test error per mask, on the same rows, so states can be sliced out below.
                errors = {}
                for arm in masks.BASE_ARMS:
                    log.info(f"{crop} month={month} model={model_name} arm={arm}")
                    predicted = masks.predict(
                        train,
                        test,
                        masks.feature_columns(arm),
                        seeds,
                        model_name,
                    )
                    errors[arm] = (actual - predicted) ** 2

                # q = RMSE(mask) / RMSE(cropland) on this state's rows; below 1 means the mask is better.
                for state, state_rows in test.groupby("state_abbr"):
                    positions = test.index.get_indexer(state_rows.index)
                    counties = test.iloc[positions]["county_fips"].to_numpy()
                    reference = errors["cropland"][positions]
                    for arm in masks.BASE_ARMS:
                        arm_error = errors[arm][positions]
                        q = float(np.sqrt(arm_error.mean() / reference.mean()))
                        # Cropland is the reference, so it has no interval.
                        if arm == "cropland":
                            low = high = np.nan
                        else:
                            # 95% interval from resampling counties.
                            low, high = masks.bootstrap_q(
                                arm_error,
                                reference,
                                counties,
                                args.bootstrap,
                            )
                        rows.append({
                            "crop": crop,
                            "cutoff_month": month,
                            "model": model_name,
                            "state": state,
                            "arm": arm,
                            "n": len(positions),
                            "n_counties": state_rows["county_fips"].nunique(),
                            "test_years": f"{test_years[0]}-{test_years[-1]}",
                            "rmse": float(np.sqrt(arm_error.mean())),
                            "q_vs_cropland": q,
                            "q_vs_cropland_lo": low,
                            "q_vs_cropland_hi": high,
                        })

    output = masks.OUT_DIR / f"masks_by_state{suffix}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    log.info(f"Wrote {output}")


if __name__ == "__main__":
    main()
