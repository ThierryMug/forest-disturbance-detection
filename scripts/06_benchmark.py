"""Runtime and storage over nested subsets of real, eligible records.

Subsets are prefixes of one seeded permutation of the eligible cell_ids, so smaller
subsets are contained in larger ones and every feature set sees identical records.
No duplicated or synthetic rows are added. Sizes above the available row count are skipped.

Preparation: Spark start-up timed once, separately; one untimed warm-up run; then each
size is timed `repeats` times (CSV read -> validation -> dedup -> join -> changes ->
Parquet write; the write is the action that forces the lazy plan). No caching.
Inference: first-seed model per feature set, fixed batch size, one warm-up call, then
`repeats` timed predict + reconstruction-error calls. Training time is reported by 04_train.
"""
import shutil
import time

import keras
import numpy as np
import pandas as pd

from fdd import plots
from fdd.config import cli_config, dir_size_bytes, get_path, load_json, save_json
from fdd.envinfo import environment_info
from fdd.models import Scaler, reconstruction_error
from fdd.schema import FEATURE_SETS
from fdd.spark_prep import prepare, timed_spark_startup


def add_args(p):
    p.add_argument("--skip-prep", action="store_true")
    p.add_argument("--skip-inference", action="store_true")


def summarize(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    out = df.groupby(keys, sort=False)[value].agg(["median", "min", "max", "count"]).reset_index()
    return out.rename(columns={c: f"{value}_{c}" for c in ["median", "min", "max", "count"]})


cfg, args = cli_config(__doc__, add_args)
b = cfg["benchmark"]
raw, proc, bench = get_path(cfg, "raw_dir"), get_path(cfg, "processed_dir"), get_path(cfg, "bench_dir")
models_dir, res = get_path(cfg, "models_dir"), get_path(cfg, "results_dir")
out_dir = res / "benchmarks"
out_dir.mkdir(parents=True, exist_ok=True)

prepared = pd.read_parquet(proc / "prepared.parquet")
ids = np.sort(prepared["cell_id"].to_numpy())
ids = ids[np.random.default_rng(cfg["project"]["seed"]).permutation(len(ids))]
sizes = [n for n in b["sizes"] if n <= len(ids)]
skipped = [n for n in b["sizes"] if n > len(ids)]
if skipped:
    print(f"Skipping sizes {skipped}: only {len(ids):,} eligible rows")

# ---------------------------------------------------------------- nested subsets + storage
features_raw = pd.read_csv(raw / "features.csv", dtype=str, keep_default_na=False)
labels_raw = pd.read_csv(raw / "labels.csv", dtype=str, keep_default_na=False)
subsets, storage = {}, []
for n in sizes:
    keep = set(ids[:n])
    d = bench / f"n{n}"
    d.mkdir(parents=True, exist_ok=True)
    features_raw[features_raw["cell_id"].isin(keep)].to_csv(d / "features.csv", index=False)
    labels_raw[labels_raw["cell_id"].isin(keep)].to_csv(d / "labels.csv", index=False)
    subsets[n] = d

# ---------------------------------------------------------------- preparation
if not args.skip_prep:
    spark, startup_s = timed_spark_startup(cfg)
    env = environment_info(spark)
    warm = subsets[sizes[0]]
    prepare(spark, warm / "features.csv", warm / "labels.csv", warm / "warmup.parquet", cfg, collect_report=False)

    runs = []
    for n in sizes:
        d = subsets[n]
        for rep in range(b["repeats"]):
            target = d / "prepared.parquet"
            if target.exists():
                shutil.rmtree(target)
            t0 = time.perf_counter()
            prepare(spark, d / "features.csv", d / "labels.csv", target, cfg, collect_report=b["include_prep_report"])
            runs.append({"rows": n, "repeat": rep, "seconds": time.perf_counter() - t0})
            print(f"prep n={n:>6,} rep={rep}: {runs[-1]['seconds']:.2f} s")
        storage.append({
            "rows": n,
            "features_csv_bytes": dir_size_bytes(d / "features.csv"),
            "labels_csv_bytes": dir_size_bytes(d / "labels.csv"),
            "parquet_bytes": dir_size_bytes(d / "prepared.parquet"),
            "parquet_files": len(list((d / "prepared.parquet").glob("*.parquet"))),
            "parquet_partitions_setting": cfg["spark"]["parquet_partitions"],
            "parquet_columns": len(spark.read.parquet(str(d / "prepared.parquet")).columns),
        })
    spark.stop()

    prep_runs = pd.DataFrame(runs)
    prep_runs.to_csv(out_dir / "prep_runs.csv", index=False)
    prep_summary = summarize(prep_runs, ["rows"], "seconds")
    prep_summary.to_csv(out_dir / "prep_summary.csv", index=False)
    storage = pd.DataFrame(storage)
    storage["csv_bytes"] = storage["features_csv_bytes"] + storage["labels_csv_bytes"]
    storage["parquet_to_csv_ratio"] = storage["parquet_bytes"] / storage["csv_bytes"]
    storage.to_csv(out_dir / "storage.csv", index=False)
    save_json({"spark_startup_seconds": startup_s, "caching": "none", "warmup_runs": 1,
               "include_prep_report": b["include_prep_report"], "environment": env}, out_dir / "prep_meta.json")
    plots.runtime_vs_rows(prep_summary, "seconds", None, "Preparation time (median, min-max)", "Seconds (log)",
                          out_dir / "prep_runtime.png")

# ---------------------------------------------------------------- inference
if not args.skip_inference:
    seed = cfg["project"]["model_seeds"][0]
    bs = b["inference_batch_size"]
    by_id = prepared.set_index("cell_id")
    runs = []
    for fs in FEATURE_SETS:
        if not (models_dir / fs / f"seed{seed}.keras").exists():
            continue
        model = keras.models.load_model(models_dir / fs / f"seed{seed}.keras")
        scaler = Scaler.load(models_dir / fs / "scaler.json")
        reconstruction_error(model, scaler.transform(by_id.loc[ids[:bs]]), bs)  # warm-up
        for n in sizes:
            x = scaler.transform(by_id.loc[ids[:n]])
            for rep in range(b["repeats"]):
                t0 = time.perf_counter()
                reconstruction_error(model, x, bs)
                runs.append({"set": fs, "rows": n, "repeat": rep, "seconds": time.perf_counter() - t0,
                             "n_features": x.shape[1], "n_params": int(model.count_params())})
            print(f"infer {fs} n={n:>6,}: median {np.median([r['seconds'] for r in runs[-b['repeats']:]]):.3f} s")
    infer_runs = pd.DataFrame(runs)
    infer_runs.to_csv(out_dir / "inference_runs.csv", index=False)
    infer_summary = summarize(infer_runs, ["set", "rows"], "seconds")
    infer_summary["rows_per_second_median"] = infer_summary["rows"] / infer_summary["seconds_median"]
    infer_summary.to_csv(out_dir / "inference_summary.csv", index=False)
    save_json({"model_seed": seed, "batch_size": bs, "warmup_calls": 1, "environment": environment_info()},
              out_dir / "inference_meta.json")
    plots.runtime_vs_rows(infer_summary, "seconds", "set", "Inference time (median, min-max)", "Seconds (log)",
                          out_dir / "inference_runtime.png")

print(f"Saved benchmark results to {out_dir}")
