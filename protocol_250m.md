# Protocol: 250 m vegetation analysis

This protocol sets the rules, tests and reporting plan for the 250 m analysis.

## Why switch from 1 km to 250 m

The crop mask keeps a MODIS pixel if the Cropland Data Layer (CDL) says it is
the target crop. A 1 km pixel is about 927 m across, so it often holds several
fields. So the first step was a pixel purity check: how much of the kept
pixels is really the target crop?

Before measuring, I set a pass rule. The same-year crop mask passes only if
its pixels are at least 50% target crop, at least 15 percentage points more
than generic cropland, and ahead of generic cropland in every state.

Result for 440 county-years per crop (10 counties in each of 11 states,
2022-2025):

| Share of target crop | corn, 1 km | corn, 250 m | soybeans, 1 km | soybeans, 250 m |
| --- | --- | --- | --- | --- |
| same-year crop mask | 0.440 fail | 0.687 pass | 0.418 fail | 0.665 pass |
| generic cropland | 0.301 | 0.339 | 0.293 | 0.335 |
| difference | 0.139 fail | 0.348 pass | 0.125 fail | 0.330 pass |
| previous-year crop mask | 0.268 | 0.215 | 0.246 | 0.191 |

The 1 km purity was too low, so I switched to the finest MODIS vegetation
product that covers the whole study period: MOD13Q1, with pixels about 232 m
across (called 250 m). Finer products either have more cloud gaps (Landsat)
or shorter records (Sentinel-2 and HLS). I ran the 1 km check first, then tried Iowa 2023 at 250 m
(0.743), then ran the full check. At 250 m the mask passed, and I did not
change any threshold.

The rules below were then set for the 250 m analysis. A cleaner mask does not
by itself mean better yield predictions; that is what the analysis tests.

## Rules

1. Use MODIS MOD13Q1 NDVI and EVI (250 m, every 16 days), keeping only pixels
   rated good or marginal.
2. Keep the same masks: generic cropland is CDL classes 1-60, 66-77 and
   204-254; corn is class 1 and soybeans class 5. The previous-year mask uses
   the year before.
3. Each 16-day image counts toward the month its 16-day window ends in. For
   example, July 28 to August 12 counts as August. This keeps any image taken
   after a cutoff out of that cutoff.
4. Keep everything else the same as the 1 km analysis: the growing seasons,
   the June to September cutoffs, the features (minimum, mean and maximum NDVI
   and EVI), the two models, the random seeds, the 2022-2025 future-year test,
   the five-fold unseen-county test, and the bootstrap. No model setting is
   re-tuned.
5. When comparing 1 km and 250 m, use only the county-years both have.

## What I will report

- All results for both crops, all four cutoffs, both models and both tests,
  whichever way they turn out.
- If the benefit starts in a different month, I will say so, since that
  changes the main timing finding.
- No new pass or fail threshold after seeing the results.
- The 1 km results and the failed 1 km check stay in the paper.

## One caution

The 1 km product (MOD13A3) is monthly, and the 250 m product (MOD13Q1) is
16-day, so they differ in more than pixel size. Differences between them are
reported as differences between products, not as an effect of pixel size
alone.
