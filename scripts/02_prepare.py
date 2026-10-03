"""PySpark preparation: validate, deduplicate, join, compute changes, write Parquet.

Inputs : <raw_dir>/features.csv, <raw_dir>/labels.csv  (Earth Engine exports)
Outputs: <processed_dir>/prepared.parquet, <results_dir>/prepare_report.json
"""
import time

from fdd.config import cli_config, dir_size_bytes, get_path, save_json
from fdd.envinfo import environment_info
from fdd.spark_prep import prepare, timed_spark_startup

cfg, _ = cli_config(__doc__)
raw, proc, res = get_path(cfg, "raw_dir"), get_path(cfg, "processed_dir"), get_path(cfg, "results_dir")
features_csv, labels_csv = raw / "features.csv", raw / "labels.csv"
out = proc / "prepared.parquet"

spark, startup_s = timed_spark_startup(cfg)
t0 = time.perf_counter()
report = prepare(spark, features_csv, labels_csv, out, cfg, collect_report=True)
total_s = time.perf_counter() - t0

report.update({
    "spark_startup_seconds": startup_s,
    "prepare_with_report_seconds": total_s,  # includes diagnostic counts; see 06_benchmark for controlled timings
    "features_csv_bytes": dir_size_bytes(features_csv),
    "labels_csv_bytes": dir_size_bytes(labels_csv),
    "parquet_bytes": dir_size_bytes(out),
    "parquet_files": len(list(out.glob("*.parquet"))),
    "environment": environment_info(spark),
})
save_json(report, res / "prepare_report.json")
spark.stop()

print(f"Rows: features={report['features_rows_raw']:,} labels={report['labels_rows_raw']:,} -> final={report['final_rows']:,}")
print(f"Classes: {report['final_class_counts']} | blocks: {report['final_blocks']}")
print(f"Duplicates (exact) features/labels: {report['features_exact_duplicates']}/{report['labels_exact_duplicates']}")
print(f"Unmatched: features w/o label={report['features_without_label']}, labels w/o features={report['labels_without_features']}")
print(f"CSV {(report['features_csv_bytes'] + report['labels_csv_bytes']) / 1e6:.1f} MB -> Parquet {report['parquet_bytes'] / 1e6:.1f} MB")
print(f"Saved {out} and {res / 'prepare_report.json'}")
