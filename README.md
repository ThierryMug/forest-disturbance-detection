# Scalable Forest Disturbance Detection

IT741A Big Data Programming project. Does enriching an autoencoder's inputs beyond NDVI
change improve agreement with mapped 2024 tree-cover loss (Hansen GFC v1.13) in a ~30 × 30 km
area of Värmland, and how do preparation and inference costs grow with the number of records?

```
Earth Engine (remote)                 Local
───────────────────────               ─────────────────────────────────────────────────────
S2 SR + cloud prob → mask → seasonal  features.csv ─┐
median → 30 m grid → NDVI/NDMI/NBR                   ├─ PySpark: validate, dedup, join,
Hansen → class fractions → label      labels.csv ───┘   changes → prepared.parquet
stratified sample → asset → 2 CSVs                         │
                                                    block splits (+buffer) → baseline,
                                                    AE A/B/C (Keras) → test metrics,
                                                    runtime/storage benchmarks
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), **Java 17** (for Spark), an Earth Engine-enabled
Google Cloud project.

```bash
uv sync
```

`uv.lock` pins every package version (Python 3.11, PySpark 3.5, TensorFlow/Keras 3). Run all
commands with `uv run`.

**Spark on Windows** needs Java 17 on `PATH`/`JAVA_HOME`, and writing Parquet also needs Hadoop's
`winutils.exe` and `hadoop.dll` (Hadoop 3.3.x builds) in `%HADOOP_HOME%\bin`, with that folder on
`PATH`. WSL2 (Ubuntu) avoids the winutils step entirely and is the easier option.

**Earth Engine**: run `uv run earthengine authenticate` once, then set `gee.project` (and
`gee.asset_root` for the asset export) in `config/config.yaml`.

## Execution order

| Step | Command | Output |
|---|---|---|
| 0 Feasibility | `uv run python scripts/00_feasibility.py` | `results/feasibility.json` (imagery, eligible cells, loss per block) |
| 1a Sample → asset | `uv run python scripts/01_extract.py --stage asset --wait` | EE table asset |
| 1b Asset → CSVs | `uv run python scripts/01_extract.py --stage csv --wait` | `varmland_features.csv`, `varmland_labels.csv` in Drive |
| — | Download both into `data/raw/` as `features.csv` and `labels.csv` | |
| 2 Prepare | `uv run python scripts/02_prepare.py` | `data/processed/prepared.parquet`, `results/prepare_report.json` |
| 3 Split | `uv run python scripts/03_split.py` | `data/processed/splits.parquet`, `results/splits_report.json` |
| 4 Train | `uv run python scripts/04_train.py` | `models/`, `results/tuning.csv`, `results/training.csv` |
| 5 Evaluate | `uv run python scripts/05_evaluate.py` | `results/metrics_*.csv`, `results/comparisons.csv`, `results/figures/` |
| 6 Benchmark | `uv run python scripts/06_benchmark.py` | `results/benchmarks/` |

Every script takes `--config <file>`. Tests: `uv run pytest`.

**Smoke test without Earth Engine** (synthetic data, tests the code only, not a result to report):

```bash
uv run python scripts/make_smoke_data.py --config config/smoke.yaml
```

then run steps 2–6 with `--config config/smoke.yaml`.

## Method decisions (as implemented)

**Grid.** 30 m cells in SWEREF99 TM (EPSG:3006). Hansen's native 0.00025° pixels are about
28 × 14 m at 60° N, so they are not used directly as the record grid. `cell_id = "<row>_<col>"`,
`block_id = "<brow>_<bcol>"` for 3 km blocks.

**Imagery.** Sentinel-2 L2A harmonized, June–August 2023 (before) and 2025 (after). Scenes with
`CLOUDY_PIXEL_PERCENTAGE ≥ 80` are skipped. Pixels are masked when s2cloudless probability ≥ 40
or SCL ∈ {0 no data, 1 saturated, 3 shadow, 6 water, 8/9 cloud, 10 cirrus, 11 snow}. Per-band
seasonal median at 10 m (20 m SWIR resampled nearest-neighbour by Earth Engine), then an
area-weighted mean onto the 30 m grid; NDVI = (B8−B4)/(B8+B4), NDMI = (B8−B11)/(B8+B11),
NBR = (B8−B12)/(B8+B12) are computed from the aggregated reflectance. `obs_*` is the mean number
of valid observations per cell; cells need ≥ 3 in both periods.

**Labels.** Hansen pixels are aggregated to fractions per cell. Candidate forest: mean
`treecover2000 ≥ 30` and entirely land. **Loss (1)**: ≥ 50 % of the cell mapped as lossyear 24
and no other loss year. **Stable (0)**: the whole cell is lossyear 0. Everything else is
excluded. These are agreement labels against Hansen, not ground truth; `treecover2000` is a
historical proxy.

**Sampling.** `stratifiedSample` with up to 30 000 stable and 8 000 loss cells (all loss cells if
fewer). The loss class is therefore **enriched**: report sampled prevalence next to precision.

**Preparation (PySpark).** CSVs are read as strings, header-checked and cast by name (cast
failures counted as type errors); range rules drop out-of-range rows; exact duplicates are
dropped and ids that still conflict are removed; features and labels are inner-joined on
`cell_id`; rows missing required values are dropped; `d_<index> = after − before`. Output is
Parquet with a fixed number of partitions.

**Splits.** Whole 3 km blocks go to train/val/test (target 60/20/20 of cells *and* of loss
cells, with per-split minimums). Cells within 60 m of a neighbouring block of a different split
are dropped. Splits are fixed before fitting and shared by all methods.

**Models.**

| Method | Inputs | Score |
|---|---|---|
| Baseline | dNDVI | −dNDVI (larger decrease = more likely loss) |
| AE-A | dNDVI | reconstruction MSE |
| AE-B | dNDVI, dNDMI, dNBR | reconstruction MSE |
| AE-C | before + after NDVI/NDMI/NBR + 3 changes (9) | reconstruction MSE |

One dense template: input → 16 ReLU → latent (n_features // 3, min 1) → 16 ReLU → output, with
Gaussian input noise during training (denoising). The noise matters for AE-A: a 1-input
autoencoder with a 1-unit bottleneck could otherwise learn the identity and score everything
near zero. Scaler and autoencoders are fitted on **stable training cells only**; early stopping
uses stable validation loss. Tuning grid (hidden units × noise) is chosen by validation average
precision; each model's threshold maximizes validation F1. Test blocks are only used in step 5.

Note the scores are not symmetric: the baseline only flags NDVI *decreases*, while reconstruction
error also flags unusual increases. Keep this in mind when interpreting A vs. baseline.

**Evaluation.** Precision, recall, F1 (at the validation threshold), average precision, PR
curves, confusion matrices, score distributions and outcome maps on test blocks. Model
differences (B−A, C−B, A−baseline, C−baseline) get 95 % intervals from a paired bootstrap that
resamples **test blocks**, averaged over model seeds.

**Benchmarks.** Nested subsets (2 000 / 8 000 / 20 000 eligible real rows; larger sizes skipped
if unavailable). Spark start-up is timed once separately; one warm-up run; each size is timed 3
times (median and range) with no caching. The timed preparation excludes diagnostic count
actions (`benchmark.include_prep_report: false`) because each count is an extra Spark job.
Inference uses the first-seed model per feature set with batch size 1024. At these sizes,
fixed overheads (JVM jobs, Keras `predict` set-up) are expected to dominate; that is a result to
report, not a flaw to hide. Training time is recorded separately in `results/training.csv`.

## Repository layout

```
config/       config.yaml (all parameters), smoke.yaml (synthetic test overrides)
src/fdd/      gee.py, spark_prep.py, splits.py, models.py, metrics.py, plots.py, envinfo.py
scripts/      00–06 pipeline steps, make_smoke_data.py
docs/         project_log.md (problems, changes, departures), architecture.md (larger-scale design)
tests/        unit tests for splits, metrics, model template
```

`data/`, `models/` and `results/` are generated and git-ignored.

## References

1. Google. Harmonized Sentinel-2 MSI Level-2A (SR). Earth Engine Data Catalog. `COPERNICUS/S2_SR_HARMONIZED`
2. Google. Sentinel-2: Cloud Probability. Earth Engine Data Catalog. `COPERNICUS/S2_CLOUD_PROBABILITY`
3. Google. Hansen Global Forest Change v1.13 (2000–2025). Earth Engine Data Catalog. `UMD/hansen/global_forest_change_2025_v1_13`
4. Hansen, M. C., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850–853. https://doi.org/10.1126/science.1244693
