"""PySpark preparation: CSV -> validated, deduplicated, joined, enriched Parquet.

The pipeline is lazy until the Parquet write (the only action when
collect_report=False), so timing prepare() includes every transformation.
No caching is used; diagnostic counts (collect_report=True) re-read the input.
"""
from __future__ import annotations

import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .schema import AFTER, BEFORE, CHANGE, FEATURE_CSV_COLS, INDEXES, LABEL_CSV_COLS, OBS

STRING_COLS = {"cell_id", "block_id"}


def build_spark(cfg: dict) -> SparkSession:
    s = cfg["spark"]
    spark = (
        SparkSession.builder.master(s["master"])
        .appName("forest-disturbance-prep")
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def timed_spark_startup(cfg: dict) -> tuple[SparkSession, float]:
    """Session creation plus one trivial job, so JVM/executor start-up is measured once."""
    t0 = time.perf_counter()
    spark = build_spark(cfg)
    spark.range(1).count()
    return spark, time.perf_counter() - t0


def _schema(cols: list[str]) -> list[tuple[str, T.DataType]]:
    return [(c, T.StringType() if c in STRING_COLS else T.DoubleType()) for c in cols]


def read_typed(spark: SparkSession, path: str | Path, cols: list[str]) -> tuple[DataFrame, DataFrame]:
    """Read CSV as strings, check the header, cast by name.

    Returns (typed_df, type_error_counts_df). A type error is a non-empty value that
    fails to cast. Reading as strings makes the check independent of column order.
    """
    raw = spark.read.option("header", True).csv(str(path))
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {raw.columns}")

    typed, errors = [], []
    for name, dtype in _schema(cols):
        r = F.col(name)
        c = F.trim(r).cast(dtype)
        typed.append(c.alias(name))
        if not isinstance(dtype, T.StringType):
            bad = r.isNotNull() & (F.trim(r) != "") & c.isNull()
            errors.append(F.sum(bad.cast("int")).alias(name))
    return raw.select(*typed), raw.agg(*errors)


# Range rules. A rule is violated only when the value is present and out of range;
# missing values are handled after the join so they can be reported per column.
FEATURE_RULES = {
    **{c: (-1.0, 1.0) for c in BEFORE + AFTER},
    **{c: (0.0, None) for c in OBS},
    "lon": (-180.0, 180.0),
    "lat": (-90.0, 90.0),
}
LABEL_RULES = {
    "loss_frac": (0.0, 1.0),
    "stable_frac": (0.0, 1.0),
    "other_frac": (0.0, 1.0),
    "land_frac": (0.0, 1.0),
    "treecover": (0.0, 100.0),
}


def _violation(col: str, lo, hi):
    c = F.col(col)
    cond = F.isnan(c)
    if lo is not None:
        cond = cond | (c < lo)
    if hi is not None:
        cond = cond | (c > hi)
    return c.isNotNull() & cond


def apply_rules(df: DataFrame, rules: dict, extra: dict | None = None) -> tuple[DataFrame, dict]:
    """Drop rows violating any rule. Returns (clean_df, {rule_name: violation_expr})."""
    exprs = {name: _violation(name, lo, hi) for name, (lo, hi) in rules.items()}
    exprs.update(extra or {})
    any_bad = F.lit(False)
    for e in exprs.values():
        any_bad = any_bad | e
    return df.filter(~any_bad), exprs


def deduplicate(df: DataFrame) -> DataFrame:
    """Drop exact duplicate rows, then drop every row of an id that still conflicts."""
    exact = df.dropDuplicates()
    conflicting = exact.groupBy("cell_id").count().filter("count > 1").select("cell_id")
    return exact.join(conflicting, "cell_id", "left_anti")


def prepare(
    spark: SparkSession,
    features_csv: str | Path,
    labels_csv: str | Path,
    out_parquet: str | Path,
    cfg: dict,
    collect_report: bool = True,
) -> dict:
    min_obs = cfg["gee"]["min_valid_obs"]
    n_parts = cfg["spark"]["parquet_partitions"]

    feats, feat_type_err = read_typed(spark, features_csv, FEATURE_CSV_COLS)
    labs, lab_type_err = read_typed(spark, labels_csv, LABEL_CSV_COLS)

    label_bad = F.col("label").isNotNull() & ~F.col("label").isin(0.0, 1.0)
    feats_ok, feat_rules = apply_rules(feats.filter(F.col("cell_id").isNotNull()), FEATURE_RULES)
    labs_ok, lab_rules = apply_rules(labs.filter(F.col("cell_id").isNotNull()), LABEL_RULES, {"label": label_bad})

    feats_d = deduplicate(feats_ok)
    labs_d = deduplicate(labs_ok)

    joined = feats_d.join(labs_d, "cell_id", "inner")
    required = BEFORE + AFTER + OBS + ["label", "x", "y", "block_id"]
    complete = joined.dropna(subset=required)
    quality = complete.filter((F.col("obs_before") >= min_obs) & (F.col("obs_after") >= min_obs))

    out = quality.withColumn("label", F.col("label").cast("int"))
    for idx in INDEXES:
        out = out.withColumn(f"d_{idx}", F.col(f"{idx}_after") - F.col(f"{idx}_before"))

    out.repartition(n_parts).write.mode("overwrite").parquet(str(out_parquet))

    if not collect_report:
        return {}

    def row_dict(df):
        return {k: int(v or 0) for k, v in df.first().asDict().items()}

    report = {
        "features_rows_raw": feats.count(),
        "labels_rows_raw": labs.count(),
        "features_type_errors": row_dict(feat_type_err),
        "labels_type_errors": row_dict(lab_type_err),
        "features_rule_violations": row_dict(feats.agg(*[F.sum(e.cast("int")).alias(k) for k, e in feat_rules.items()])),
        "labels_rule_violations": row_dict(labs.agg(*[F.sum(e.cast("int")).alias(k) for k, e in lab_rules.items()])),
        "features_rows_after_rules": feats_ok.count(),
        "labels_rows_after_rules": labs_ok.count(),
        "features_exact_duplicates": feats_ok.count() - feats_ok.dropDuplicates().count(),
        "labels_exact_duplicates": labs_ok.count() - labs_ok.dropDuplicates().count(),
        "features_rows_after_dedup": feats_d.count(),
        "labels_rows_after_dedup": labs_d.count(),
        "features_without_label": feats_d.join(labs_d, "cell_id", "left_anti").count(),
        "labels_without_features": labs_d.join(feats_d, "cell_id", "left_anti").count(),
        "joined_rows": joined.count(),
        "missing_after_join": row_dict(joined.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in required])),
        "rows_after_missing_filter": complete.count(),
    }
    final = spark.read.parquet(str(out_parquet))
    report["final_rows"] = final.count()
    report["final_class_counts"] = {str(r["label"]): int(r["count"]) for r in final.groupBy("label").count().collect()}
    report["final_blocks"] = final.select("block_id").distinct().count()
    report["final_columns"] = final.columns
    report["change_columns"] = CHANGE
    return report
