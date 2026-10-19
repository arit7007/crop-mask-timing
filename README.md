# Crop-Mask Timing Study

Code and data for the study *When Does Knowing Where Corn and Soybeans Grow
Improve County-Level Yield Prediction, and Is Last Year's Map Enough?*

This study tests when crop maps help predict county-level corn and soybean
yields, and whether last year's map can provide similar benefits. It covers
11 states from 2009 to 2025, with predictions at June, July, August and
September cutoffs.

A mask selects the satellite pixels used in each county's vegetation summary.
The three main masks are:

- generic cropland from the same year's Cropland Data Layer (CDL),
- the target crop's final same-year CDL map,
- the target crop's CDL map from the year before.

The models are random forest and histogram gradient boosting. Within each
comparison, the masks use the same county-years, weather and soil inputs, and
model settings. Only the vegetation summaries change. The two tests are:

- **Future years:** train on 2009-2021 and test on 2022-2025.
- **Unseen counties:** hold out whole counties in five groups, so a county is
  never in both training and testing. This test shares calendar years with the
  training data.

Separate runs compare last year's generic cropland and a simple rule that
swaps last year's corn and soybean locations. The rotation rule is tested on
future years only.

The research paper describes the findings and limitations. Both same-year
masks use the final CDL, released after the growing season. The study measures
the value of knowing crop locations; it does not demonstrate a forecast using
only information available at the prediction date.

## Install

You need Python 3.14 or later and `uv`. Run all commands from the repository
root. Install the locked package versions with:

```powershell
uv sync --frozen
```

Checking saved results and refitting from the included data do not need a
NASS key or an Earth Engine account.

## Quick start

Check the saved results without downloading data or refitting models:

```powershell
uv run python src/run_study.py --quick
```

Refit the models and rerun the main analysis using the included data, without
downloading from the original sources:

```powershell
tar -xkf data/inputs_1km.zip
tar -xkf data/inputs_250m.zip
uv run python src/merge.py
uv run python src/compare_masks_by_month.py --product MOD13A3 --table-only
uv run python src/run_study.py --validate
```

On macOS or Linux, use `unzip -n` instead of `tar -xkf`. These commands keep
existing input files; check them against the checksums in [Inputs](#inputs).

`merge.py` joins the processed tables. The `--table-only` command prepares the
1 km comparison without fitting models. The final command fits the 250 m
models, rebuilds outputs, runs the unit checks, and independently refits the
main late-season corn results. It uses local files and can take a long time.

Use `uv run python src/run_study.py` to run the analysis without the additional
validation. `--quick --validate` checks saved validation reports without
rerunning those checks.

## Inputs

The monthly inputs and crop-progress data used for the paper are included in:

- `data/inputs_1km.zip`: monthly climate, 1 km generic-cropland vegetation and
  weekly crop progress,
- `data/inputs_250m.zip`: the 250 m vegetation files, plus the 1 km
  crop-specific vegetation files.

`data/inputs_sha256.txt` lists their checksums. In Git Bash or Linux, check the
extracted files with:

```bash
sha256sum -c data/inputs_sha256.txt
```

Every line should say OK. On macOS, use
`shasum -a 256 -c data/inputs_sha256.txt`. In PowerShell, use
`Get-FileHash <file>` and compare its SHA-256 value with the listed checksum.

`data/processed/` holds the yield, climate, vegetation and soil tables.
`merge.py` joins them into `merged.csv`, and the analysis rebuilds the climate
and vegetation features from the monthly files at each cutoff. The merged and
feature tables are rebuilt, so they are not kept in Git.

The reported results use MOD13Q1, the 16-day 250 m MODIS product. MOD13A3, the
monthly 1 km product, is a secondary comparison. File names with `_250m` refer
to MOD13Q1. `protocol_250m.md` records the processing rules and crop-purity
test. The test years are the last four years in the input data, so adding
years changes the test set.

The runner includes the rotation test. It also runs the previous-year
cropland and matched-product comparisons when their inputs are available,
and reports when either is skipped.

## Validation

| Program | What it checks |
| --- | --- |
| `validation/check_units.py` | Input ranges, missing values, temperature ordering, and yield units. |
| `validation/verify_results.py` | Independently refits August and September corn results and checks all 24 comparisons against the saved results. |
| `validation/verify_masks.py` | Recalculates county NDVI in Earth Engine for three counties per state and compares it with the saved summaries for the three main masks. |

The first two run with `run_study.py --validate`. The Earth Engine check runs
separately and needs an Earth Engine account:

```powershell
uv run python validation/verify_masks.py --product MOD13Q1
```

It checks 2022-2025 by default. For a smaller check, add `--years 2023 2025`.
Include 2022 or 2023 for the strict 30 m CDL check. A run replaces
`results/verify_masks_250m.csv`.

## Results

Outputs are written to `results/`:

| File | What it holds | Made by |
| --- | --- | --- |
| `masks_by_month*.csv` | Main comparison by crop, cutoff, model, test and arm | `compare_masks_by_month.py` |
| `masks_by_month*_by_year.csv` | The same comparison for each test year | `compare_masks_by_month.py` |
| `masks_by_month_250m_matched*.csv` | Main comparison on the county-years both products share | `compare_masks_by_month.py --match-product` |
| `masks_by_state_250m.csv` | Results by state | `compare_masks_by_state.py` |
| `masks_rotation_250m*.csv` | Rotation-rule test | `compare_masks_by_month.py --include-rotation-proxy` |
| `masks_prev_year_cropland_250m*.csv` | Previous-year cropland arm | `compare_masks_by_month.py --include-prev-year-cropland` |
| `two_mask_contrast_250m.csv` | Previous-year crop mask against previous-year cropland | `compare_two_masks.py` |
| `harvest_timing_250m.csv` | August and September results limited by harvest progress | `check_harvest_timing.py` |
| `mask_purity*.csv` | Share of target crop in the MODIS pixels each mask keeps | `measure_mask_purity.py` |
| `verify_results_250m.csv` | Independent refit of the August and September corn results | `verify_results.py` |
| `verify_masks_250m.csv` | Independent rebuild of the county means from the CDL | `verify_masks.py` |
| `unit_checks_250m.csv` | Unit and range checks on the model inputs | `check_units.py` |

The figures, purity summaries and input-check reports are ignored by Git.
Rebuild the figures and purity summaries from saved results with:

```powershell
uv run python src/make_figures.py
uv run python src/measure_mask_purity.py --product MOD13A3 --summarize-only
uv run python src/measure_mask_purity.py --summarize-only
uv run python src/measure_mask_purity.py --summarize-only --all-counties
```

In the CSVs, `arm` identifies the mask and `q_vs_cropland` is its RMSE divided
by generic-cropland RMSE. Lower is better: 0.95 means 5% lower RMSE. RMSE is a
prediction-error measure that gives more weight to larger errors.

## Download from sources

This is optional. To download source data, copy `.env.example` to `.env` and
set `NASS_API_KEY` and `GEE_PROJECT_ID`. The NASS key is needed for yield and
crop-progress downloads. Earth Engine is needed for vegetation, weather, soil,
the live mask check, and live purity measurements. Run
`earthengine authenticate` once for Earth Engine. Keep `.env` out of Git.

The data providers sometimes update their data, so new downloads may not match
the included files.

Yield, climate, vegetation and soil tables:

```powershell
uv run python src/download_yield.py --study --start_year 2008 --end_year 2025
uv run python src/download_climate.py --study --years_start 2008 --years_end 2025
uv run python src/download_vegetation.py --study --start_year 2008 --end_year 2025
uv run python src/download_soil.py --study
```

Crop-specific 1 km files and crop progress:

```powershell
uv run python src/download_vegetation.py --study --start_year 2009 --end_year 2025 --variant crop_specific --crop-specific-mask --mask-crops CORN SOYBEANS
uv run python src/download_vegetation.py --study --start_year 2009 --end_year 2025 --variant crop_specific_prev_year --crop-specific-mask --mask-crops CORN SOYBEANS --mask-year-offset -1
uv run python src/download_crop_progress.py --start-year 2009 --end-year 2025
```

The 250 m files, which the reported results use:

```powershell
uv run python src/download_vegetation.py --study --start_year 2009 --end_year 2025 --product MOD13Q1 --variant 250m
uv run python src/download_vegetation.py --study --start_year 2009 --end_year 2025 --product MOD13Q1 --variant crop_specific_250m --crop-specific-mask --mask-crops CORN SOYBEANS
uv run python src/download_vegetation.py --study --start_year 2009 --end_year 2025 --product MOD13Q1 --variant crop_specific_prev_year_250m --crop-specific-mask --mask-crops CORN SOYBEANS --mask-year-offset -1
uv run python src/download_vegetation.py --study --start_year 2009 --end_year 2025 --product MOD13Q1 --variant prev_year_cropland_250m --mask-year-offset -1
```

Each vegetation `--variant` writes a separate file in `data/raw/`. The
previous-year maps start in 2009 because nationwide CDL coverage begins in
2008. The 250 m downloads can take hours and save progress so they can resume.
After downloading, follow [Quick start](#quick-start) from `merge.py` onward.

## What came from earlier work

Parts of the code and data pipeline were adapted from
[CropCast, release v2026.09.10](https://github.com/ScienceAndBeyond/CropCast/tree/v2026.09.10).
CropCast was first developed by Arit Prince and Arya Prince for AGU 2025 and
continued by Arit Prince in 2026.

The adapted parts include the data downloaders, shared settings and helpers,
vegetation aggregation, table merging, and parts of the random-forest setup.
The crop-mask study adds the 250 m product, the crop-specific and
previous-year masks, the rule that assigns each 16-day composite to the month
its window ends in, monthly comparisons, histogram gradient boosting, state comparisons, harvest checks,
purity measurements, and independent validation programs.

## License

MIT. See [LICENSE](LICENSE).
