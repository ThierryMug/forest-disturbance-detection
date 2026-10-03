"""Record hardware and software context for benchmark reporting."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import sys

import psutil

PACKAGES = ["pyspark", "tensorflow", "keras", "numpy", "pandas", "pyarrow", "scikit-learn", "earthengine-api"]


def environment_info(spark=None) -> dict:
    info = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": sys.version.split()[0],
        "packages": {},
    }
    for p in PACKAGES:
        try:
            info["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            info["packages"][p] = None
    if spark is not None:
        jvm = spark.sparkContext._jvm
        info["java"] = jvm.System.getProperty("java.version")
        info["spark_version"] = spark.version
        info["spark_master"] = spark.sparkContext.master
        info["spark_conf"] = {
            k: spark.conf.get(k) for k in ("spark.driver.memory", "spark.sql.shuffle.partitions")
        }
    return info
