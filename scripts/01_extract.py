"""Extract the analytical sample from Earth Engine.

Two stages, so both CSVs come from one materialized sample:
  --stage asset : stratified sample -> Earth Engine table asset
  --stage csv   : asset -> features CSV + labels CSV in Google Drive
Download the two CSVs into paths.raw_dir as features.csv and labels.csv.
Use --stage direct to skip the asset (the sample is then computed once per export).
"""
import time

import ee

from fdd import gee
from fdd.config import cli_config, get_path, save_json


def add_args(p):
    p.add_argument("--stage", choices=["asset", "csv", "direct"], required=True)
    p.add_argument("--wait", action="store_true", help="poll until the started tasks finish")


cfg, args = cli_config(__doc__, add_args)
gee.initialize(cfg)
g = cfg["gee"]
prefix = g["export"]["prefix"]
asset_id = f"{g['asset_root']}/{prefix}_sample" if g.get("asset_root") else None
feature_cols, label_cols = gee.export_selectors()
tasks = []


def to_drive(fc, name, selectors):
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"{prefix}_{name}",
        folder=g["export"]["drive_folder"],
        fileNamePrefix=f"{prefix}_{name}",
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    tasks.append(task)


if args.stage in ("asset", "direct"):
    region = gee.study_area(cfg)
    img, counts = gee.analysis_image(cfg, region)
    sample = gee.stratified_sample(cfg, img, region)
    meta = {
        "config_source": cfg["_source"],
        "gee": g,
        "seed": cfg["project"]["seed"],
        "n_images_before": counts["n_images_before"].getInfo(),
        "n_images_after": counts["n_images_after"].getInfo(),
        "asset_id": asset_id,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(meta, get_path(cfg, "raw_dir") / "extraction_meta.json")
    print(f"Images: before={meta['n_images_before']} after={meta['n_images_after']}")

    if args.stage == "asset":
        if not asset_id:
            raise SystemExit("Set gee.asset_root in the config (e.g. projects/<project>/assets).")
        task = ee.batch.Export.table.toAsset(collection=sample, description=f"{prefix}_sample", assetId=asset_id)
        task.start()
        tasks.append(task)
    else:
        to_drive(sample, "features", feature_cols)
        to_drive(sample, "labels", label_cols)

if args.stage == "csv":
    fc = ee.FeatureCollection(asset_id)
    print(f"Sample rows in asset: {fc.size().getInfo():,}")
    to_drive(fc, "features", feature_cols)
    to_drive(fc, "labels", label_cols)

for t in tasks:
    print(f"Started task {t.id}: {t.status()['description']}")

if args.wait:
    t0 = time.time()
    while any(t.status()["state"] in ("READY", "RUNNING") for t in tasks):
        time.sleep(30)
    for t in tasks:
        s = t.status()
        print(f"{s['description']}: {s['state']} {s.get('error_message', '')}")
    # Descriptive only: EE queueing and export time are not a controlled benchmark.
    print(f"Wall time waiting: {time.time() - t0:.0f} s")
