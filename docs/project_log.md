# Project log

Record unexpected problems, how they were resolved, and every departure from the proposal
(with the config change and the date). Newest entries at the top.

| Date | Area | What happened | Decision / change | Config or commit |
|---|---|---|---|---|
| 2026-10-03 | Grid | Hansen pixels are ~28 × 14 m at 60° N, not square 30 m. | Use a 30 m SWEREF99 TM grid; Hansen classes aggregated to per-cell fractions (purity rules in README). | `gee.crs`, `gee.grid_m` |
| 2026-10-03 | Model | A 1-input autoencoder with a 1-unit bottleneck can learn the identity. | Same template for all sets with Gaussian input noise (denoising); noise level in the tuning grid. | `autoencoder.noise_std` |
| 2026-10-03 | Benchmark | Diagnostic counts are separate Spark jobs and would inflate prep timings. | Timed runs exclude them; full report produced once by `02_prepare.py`. | `benchmark.include_prep_report` |

## Feasibility check (fill in after `00_feasibility.py`)

- AOI centre / size:
- Images before / after:
- Eligible loss / stable cells:
- Blocks with ≥ 20 loss cells:
- Decision (keep / replace region):

## Extraction record

- Earth Engine task IDs and states:
- Sample rows, class proportions, CSV sizes:
- Usable area and image counts (from `data/raw/extraction_meta.json`):

## What I would change in a repeat study
