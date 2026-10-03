"""Assign geographic blocks to train/val/test with a border buffer (before any model fitting).

Outputs: <processed_dir>/splits.parquet, <results_dir>/splits_report.json, figures/split_map.png
"""
import pandas as pd

from fdd import plots
from fdd.config import cli_config, get_path, save_json
from fdd.splits import make_splits

cfg, _ = cli_config(__doc__)
proc, res = get_path(cfg, "processed_dir"), get_path(cfg, "results_dir")

df = pd.read_parquet(proc / "prepared.parquet", columns=["cell_id", "x", "y", "label"]).sort_values("cell_id").reset_index(drop=True)
assignment, report = make_splits(df, cfg)
assignment.to_parquet(proc / "splits.parquet", index=False)
save_json(report, res / "splits_report.json")
plots.split_map(df.merge(assignment, on="cell_id"), res / "figures" / "split_map.png")

print(f"Blocks: {report['n_blocks']} | buffer-dropped cells: {report['cells_buffer_dropped']:,}")
for s, d in report["per_split"].items():
    print(f"  {s:5s} blocks={d['blocks']:3d} cells={d['cells']:6,d} loss={d['loss']:5,d} prevalence={d['loss_prevalence']:.3f}")
