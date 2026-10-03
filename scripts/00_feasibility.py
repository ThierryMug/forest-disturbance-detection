"""Feasibility check: Earth Engine access, usable imagery, and loss cells per block.

Run before any export. Writes results/feasibility.json and prints PASS/FAIL per check.
"""
import ee

from fdd import gee
from fdd.config import cli_config, get_path, save_json

cfg, _ = cli_config(__doc__)
gee.initialize(cfg)
g, fz = cfg["gee"], cfg["feasibility"]

region = gee.study_area(cfg)
img, counts = gee.analysis_image(cfg, region)
label = img.select("label")

reduce_kw = dict(geometry=region, crs=g["crs"], scale=g["grid_m"], maxPixels=1e10, tileScale=4)

# Eligible cells by class (after forest, purity and valid-observation filters).
class_counts = ee.Image.cat([label.eq(1).rename("loss"), label.eq(0).rename("stable")]).reduceRegion(
    ee.Reducer.sum(), **reduce_kw
)
# Loss cells per geographic block.
per_block = ee.Image.cat([label.eq(1).rename("loss"), gee.block_index(cfg)]).reduceRegion(
    ee.Reducer.sum().group(groupField=1, groupName="block"), **reduce_kw
)
# Valid-observation statistics over candidate forest cells (before quality filtering).
before, _ = gee.seasonal_features(cfg, "before", region)
after, _ = gee.seasonal_features(cfg, "after", region)
forest = gee.label_layers(cfg).select("treecover").gte(g["treecover_threshold"])
obs_stats = ee.Image.cat([before.select("obs_before"), after.select("obs_after")]).updateMask(forest).reduceRegion(
    ee.Reducer.percentile([10, 50, 90]), **reduce_kw
)

info = ee.Dictionary({
    "n_images_before": counts["n_images_before"],
    "n_images_after": counts["n_images_after"],
    "class_counts": class_counts,
    "per_block": per_block,
    "obs_percentiles": obs_stats,
    "area_km2": region.area(1).divide(1e6),
}).getInfo()

blocks = {int(d["block"]): int(round(d["sum"])) for d in info["per_block"]["groups"]}
n_loss = int(round(info["class_counts"]["loss"]))
n_stable = int(round(info["class_counts"]["stable"]))
blocks_with_loss = sum(v >= fz["min_loss_per_block"] for v in blocks.values())

checks = {
    "imagery_before": info["n_images_before"] > 0,
    "imagery_after": info["n_images_after"] > 0,
    "enough_loss_cells": n_loss >= fz["min_loss_cells"],
    "enough_blocks_with_loss": blocks_with_loss >= fz["min_blocks_with_loss"],
}
result = {
    "aoi": g["aoi"],
    "area_km2": info["area_km2"],
    "n_images_before": info["n_images_before"],
    "n_images_after": info["n_images_after"],
    "eligible_loss_cells": n_loss,
    "eligible_stable_cells": n_stable,
    "blocks_total": len(blocks),
    "blocks_with_loss": blocks_with_loss,
    "loss_cells_per_block": blocks,
    "obs_percentiles_forest": info["obs_percentiles"],
    "checks": checks,
    "pass": all(checks.values()),
}
out = get_path(cfg, "results_dir") / "feasibility.json"
save_json(result, out)

print(f"Area: {info['area_km2']:.0f} km2 | images before/after: {info['n_images_before']}/{info['n_images_after']}")
print(f"Eligible cells: loss={n_loss:,}  stable={n_stable:,}  | blocks with >= {fz['min_loss_per_block']} loss: "
      f"{blocks_with_loss}/{len(blocks)}")
print(f"Valid observations (forest, p10/p50/p90): {info['obs_percentiles']}")
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"Saved {out}")
