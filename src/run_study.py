"""Run or check the crop-mask timing analysis.

The reported results use MOD13Q1 at 250 m, so that is the default.
Use --product MOD13A3 for the 1 km analysis.

Run from the project root:
    python src/run_study.py                      # full analysis
    python src/run_study.py --quick              # check saved results
    python src/run_study.py --validate           # also run validation
    python src/run_study.py --product MOD13A3    # the earlier 1 km set
"""

import argparse
from itertools import product
import subprocess
import sys
from pathlib import Path

import pandas as pd

import compare_masks_by_month as masks


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VALIDATION_CODE = ROOT / "validation"
RESULTS = ROOT / "results"
FOURTH_ARM = "previous_year_cropland"


def run(script: str, *arguments: str) -> None:
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = ROOT / script_path
    command = [sys.executable, "-X", "utf8", str(script_path), *arguments]
    print("Running", " ".join(command[3:]))
    subprocess.run(command, cwd=SRC, check=True)


def check_inputs(suffix: str) -> bool:
    """Check the files this product needs. Returns whether the fourth arm is
    available, since it is optional."""
    vegetation = masks.veg_monthly(suffix)
    needed = [vegetation[arm] for arm in masks.BASE_ARMS] + [
        ROOT / "data/processed/merged.csv",
        ROOT / "data/raw/climate_monthly.csv",
        ROOT / "data/raw/crop_progress_weekly.csv",
    ]
    missing = [path for path in needed if not path.exists()]
    if missing:
        names = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{names}")
    return vegetation[FOURTH_ARM].exists()


def check_outputs(suffix: str, validated: bool = False) -> list[str]:
    checks = []
    results = pd.read_csv(RESULTS / f"masks_by_month{suffix}.csv")
    keys = ["crop", "cutoff_month", "model", "evaluation", "arm"]
    expected = set(product(
        ("CORN", "SOYBEANS"), (6, 7, 8, 9),
        ("random_forest", "hist_gradient_boosting"),
        ("temporal_holdout", "county_heldout_cv"),
        masks.BASE_ARMS,
    ))
    if results.duplicated(keys).any() or set(results[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Primary results are incomplete or contain duplicate comparisons")
    if not (results[["rmse", "q_vs_cropland", "n_test"]] > 0).all().all():
        raise ValueError("Primary results contain missing or invalid scores")
    checks.append(f"Saved primary comparisons: {len(results)}; all expected combinations present")

    fourth = RESULTS / f"masks_prev_year_cropland{suffix}.csv"
    if fourth.exists():
        rows = pd.read_csv(fourth)
        if FOURTH_ARM not in rows["arm"].unique():
            raise ValueError(f"{fourth.name} has no {FOURTH_ARM} arm")
        checks.append(f"Four-arm comparisons: {len(rows)}")
    contrast = RESULTS / f"two_mask_contrast{suffix}.csv"
    if contrast.exists():
        rows = pd.read_csv(contrast)
        checks.append(f"Paired arm contrasts: {len(rows)}; "
                      f"{int(rows['excludes_one'].sum())} exclude a ratio of one")

    if validated:
        for filename, label, expected_count in (
            (f"verify_results{suffix}.csv", "Independent comparisons", 24),
            (f"unit_checks{suffix}.csv", "Unit checks", 34),
        ):
            frame = pd.read_csv(RESULTS / filename)
            passed = len(frame) == expected_count and frame["passed"].astype(str).str.lower().eq("true").all()
            if not passed:
                raise ValueError(f"{label} are incomplete or contain failures")
            checks.append(f"{label}: {len(frame)}; all passed: True")
    else:
        checks.append("Validation programs were not run")

    harvest = pd.read_csv(RESULTS / f"harvest_timing{suffix}.csv")
    august = harvest[(harvest["crop"] == "CORN") & (harvest["cutoff_month"] == 8)]
    checks.append(f"August corn progress rows: {int(august['n_state_years'].max())} state-years")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Check the saved results without refitting.")
    parser.add_argument("--validate", action="store_true",
                        help="Include the independent verification and unit audit.")
    parser.add_argument("--product", choices=sorted(masks.PRODUCT_SUFFIX), default="MOD13Q1",
                        help="MODIS product to analyse. MOD13Q1 is the reported one.")
    args = parser.parse_args()
    suffix = masks.PRODUCT_SUFFIX[args.product]
    other = "MOD13A3" if args.product == "MOD13Q1" else "MOD13Q1"

    if args.validate:
        for name in ("verify_results.py", "check_units.py"):
            if not (VALIDATION_CODE / name).exists():
                parser.error(f"Validation program is missing: {name}")

    if not args.quick:
        has_fourth_arm = check_inputs(suffix)
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        run("src/compare_masks_by_month.py", "--product", args.product,
            "--log-file", str(logs / f"masks_by_month{suffix}.log"))

        other_table = ROOT / "data" / "processed" / f"features_by_cutoff{masks.PRODUCT_SUFFIX[other]}.csv"
        if other_table.exists():
            run("src/compare_masks_by_month.py", "--product", args.product,
                "--match-product", other,
                "--log-file", str(logs / f"masks_by_month{suffix}_matched.log"))
        else:
            print(f"Skipping the matched run: no {other_table.name}")
        run("src/compare_masks_by_month.py", "--product", args.product,
            "--include-rotation-proxy", "--skip-spatial",
            "--log-file", str(logs / f"masks_rotation{suffix}.log"))
        if has_fourth_arm:
            run("src/compare_masks_by_month.py", "--product", args.product,
                "--include-prev-year-cropland",
                "--log-file", str(logs / f"masks_prev_year_cropland{suffix}.log"))
            run("src/compare_two_masks.py", "--product", args.product,
                "--log-file", str(logs / f"two_mask_contrast{suffix}.log"))
        else:
            print(f"Skipping the {FOURTH_ARM} arm and the paired contrast: "
                  f"no {masks.veg_monthly(suffix)[FOURTH_ARM].name}")
        run("src/compare_masks_by_state.py", "--product", args.product)
        run("src/check_harvest_timing.py", "--product", args.product)
        run("src/make_figures.py", "--product", args.product)

        for extra in ([], ["--all-counties"]):
            name = f"mask_purity{suffix}{'_allcounties' if extra else ''}.csv"
            if (RESULTS / name).exists():
                run("src/measure_mask_purity.py", "--product", args.product,
                    "--summarize-only", *extra)
        if args.validate:
            run("validation/verify_results.py", "--product", args.product)
            run("validation/check_units.py", "--product", args.product)

    for line in check_outputs(suffix, args.validate):
        print("-", line)


if __name__ == "__main__":
    main()
