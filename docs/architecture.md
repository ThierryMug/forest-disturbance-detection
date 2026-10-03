# Architecture: prototype and larger-scale design

## Implemented prototype

| Stage | Where | Notes |
|---|---|---|
| Masking, compositing, grid alignment, sampling | Earth Engine (remote) | Bounded by EE quotas; task time is descriptive, not a controlled benchmark |
| Validation, dedup, join, change features, Parquet | PySpark `local[*]` | One machine; at 2k–50k rows Spark overhead may exceed its benefit |
| Training and inference | Keras, one machine | The training sample is **collected into local memory** (pandas/NumPy). Spark does not distribute Keras training |

## Proposed larger deployment (design discussion, not implemented)

1. **Extraction by tile and period.** Export per spatial tile (e.g. 50 km SWEREF99 TM tiles
   aligned to the 30 m grid) and season, so jobs are independent and restartable. Grid-aligned
   tiles mean no cell is split between tiles.
2. **Versioned storage.** Parquet in object storage, partitioned by `period/tile_id`
   (`s3://…/features/v=2026-10/period=2025JJA/tile=…`). Partition pruning lets inference read
   one region or period without scanning everything. Schema versions are recorded with the data.
3. **Spark cluster for preparation.** Same code as `fdd.spark_prep`, run against the partitioned
   store. Joins on `cell_id` within a tile become co-partitioned (partition both sides by
   `tile_id`) to avoid a global shuffle.
4. **Training stays on one machine** unless measurements show otherwise: train on a bounded,
   geographically representative sample (stratified by region and land-cover).
5. **Batch inference across partitions.** `mapInPandas` / `mapPartitions` with the model
   loaded **once per worker** (broadcast the model file path, load lazily); write scores with
   model version and input-data version to a versioned output table.

## Concerns to discuss

- **Partitioning**: too many small files (one per tile × period × partition) hurt both Spark and
  object storage; compact to ~128–512 MB files.
- **Memory**: driver collection (`toPandas`) is only acceptable for the bounded training
  sample; inference must stay distributed.
- **Data transfer**: EE → Drive/GCS → cluster egress costs; keep compute in the same cloud region
  as storage.
- **Spatial boundaries**: block-based evaluation and buffers must be recomputed per region;
  features needing neighbourhoods would require tile overlaps.
- **Monitoring**: track input quality (valid-observation counts, cloud share, index ranges per
  tile) and prediction drift (score distribution per tile and period versus the training
  reference); alert before trusting new outputs.

Expected gains from a cluster are hypotheses until measured; the project's benchmarks only
cover the sizes and machine recorded in `results/benchmarks/*_meta.json`.
