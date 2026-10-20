"""Compare two masks on the same county-years, with a county bootstrap interval.

The main tables compare each mask with generic cropland. This checks the
difference between two chosen masks directly.

Run: python src/compare_two_masks.py
Defaults to --numerator previous_year --denominator previous_year_cropland.
Writes results/two_mask_contrast<suffix>.csv.
"""

import argparse

import numpy as np
import pandas as pd

import compare_masks_by_month as masks


def main() -> None:
    """Compare the numerator mask's error with the denominator mask's at each cutoff."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--product", choices=sorted(masks.PRODUCT_SUFFIX), default="MOD13Q1",
                        help="MODIS product: MOD13A3 (1 km) or MOD13Q1 (250 m) "
                             "(default: %(default)s).")
    parser.add_argument("--numerator", default="previous_year",
                        help="Mask whose error is on top of the ratio "
                             f"(one of {', '.join(masks.VEG_MONTHLY)}; default: %(default)s).")
    parser.add_argument("--denominator", default="previous_year_cropland",
                        help="Mask whose error is on the bottom of the ratio (default: %(default)s).")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[8, 9],
                        help="Cutoff months to compare (default: %(default)s).")
    parser.add_argument("--bootstrap", type=int, default=masks.BOOTSTRAP_DRAWS,
                        help="County resamples for the 95%% interval (default: %(default)s).")
    parser.add_argument("--log-file", default=None,
                        help="Also write progress logs to this file.")
    args = parser.parse_args()
    masks.setup_logging(args.log_file)
    suffix = masks.select_product(args.product)

    arms = (args.numerator, args.denominator)
    vegetation, _ = masks.audit_inputs()
    missing = [a for a in arms if a not in vegetation]
    if missing:
        raise SystemExit(f"No vegetation file for {missing}")

    # The feature table always needs the base masks; add any chosen mask that isn't one.
    table_arms = tuple(masks.BASE_ARMS) + tuple(
        a for a in arms if a not in masks.BASE_ARMS)

    vegetation = {arm: vegetation[arm] for arm in table_arms}

    climate = masks.read_csv(masks.CLIMATE_MONTHLY)
    base = masks.load_base_table()

    rows = []
    for month in args.cutoffs:
        table = masks.build_cutoff_table(base, climate, vegetation, month)
        for crop in masks.CROPS:
            train, test, years = masks.temporal_split(table[table["crop"] == crop])
            actual = test["yield_value"].to_numpy()
            counties = test["county_fips"].to_numpy()
            for model in masks.MODELS:
                errors = {}
                for arm in arms:
                    predicted = masks.predict(
                        train, test, masks.feature_columns(arm), masks.SEEDS, model)
                    errors[arm] = (actual - predicted) ** 2
                # ratio = RMSE(numerator) / RMSE(denominator); below 1 means the numerator is better.
                ratio = float(np.sqrt(errors[args.numerator].mean()
                                      / errors[args.denominator].mean()))
                low, high = masks.bootstrap_q(
                    errors[args.numerator], errors[args.denominator],
                    counties, args.bootstrap)
                rows.append({
                    "crop": crop,
                    "cutoff_month": month,
                    "model": model,
                    "evaluation": "temporal_holdout",
                    "numerator": args.numerator,
                    "denominator": args.denominator,
                    "n_test": len(actual),
                    "test_years": f"{years[0]}-{years[-1]}",
                    "ratio": ratio,
                    "ratio_lo": low,
                    "ratio_hi": high,
                    # True when the 95% interval doesn't include 1, i.e. a clear difference.
                    "excludes_one": bool(low > 1 or high < 1),
                })
                masks.log.info(
                    f"{crop} cutoff={month} {model}: {ratio:.4f} [{low:.4f}, {high:.4f}]")

    result = pd.DataFrame(rows)
    path = masks.OUT_DIR / f"two_mask_contrast{suffix}.csv"
    result.to_csv(path, index=False)
    masks.log.info(f"Wrote {path}")
    print(result[["crop", "cutoff_month", "model", "ratio", "ratio_lo",
                  "ratio_hi", "excludes_one"]].to_string(index=False))


if __name__ == "__main__":
    main()
