"""Earth Engine image construction: masking, seasonal composites, labels, sampling.

Everything here builds lazy server-side objects; nothing runs until an export
or a getInfo() call. Decisions (documented in the README):

* Analysis grid: 30 m cells in cfg.gee.crs (SWEREF99 TM), aligned to the CRS origin.
* Reflectance: masked 10/20 m pixels -> seasonal per-band median -> area-weighted
  mean onto the 30 m grid -> indices computed from the aggregated reflectance.
* Labels: Hansen pixels are aggregated to class *fractions* per 30 m cell, then
  a cell is "loss" (>= loss_fraction_min mapped 2024 loss, no other loss year) or
  "stable" (entirely lossyear == 0). Mixed / other-year cells are excluded.
"""
from __future__ import annotations

import ee

from .schema import FEATURE_CSV_COLS, LABEL_CSV_COLS


def initialize(cfg: dict) -> None:
    project = cfg["gee"].get("project")
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def target_projection(cfg: dict) -> ee.Projection:
    g = cfg["gee"]
    return ee.Projection(g["crs"]).atScale(g["grid_m"])


def study_area(cfg: dict) -> ee.Geometry:
    """Square of aoi.size_m side in the analysis CRS, centred on the configured point."""
    g = cfg["gee"]
    centre = ee.Geometry.Point([g["aoi"]["center_lon"], g["aoi"]["center_lat"]])
    return centre.buffer(g["aoi"]["size_m"] / 2, 1).bounds(1, ee.Projection(g["crs"]))


# --------------------------------------------------------------------------- imagery

def _mask_s2(img: ee.Image, g: dict) -> ee.Image:
    prob = ee.Image(img.get("cloud_prob")).select("probability")
    classes = g["scl_mask_classes"]
    bad_scl = img.select("SCL").remap(classes, [1] * len(classes), 0)
    valid = prob.lt(g["cloud_prob_threshold"]).And(bad_scl.Not())
    b = g["bands"]
    refl = img.select([b["red"], b["nir"], b["swir1"], b["swir2"]], ["red", "nir", "swir1", "swir2"])
    # Harmonized collection: offsets already removed, reflectance = DN / 10000.
    return refl.divide(10000).updateMask(valid).copyProperties(img, ["system:time_start"])


def masked_collection(cfg: dict, period: str, region: ee.Geometry) -> ee.ImageCollection:
    g = cfg["gee"]
    start, end = g[period]["start"], g[period]["end"]
    sr = (
        ee.ImageCollection(g["s2_collection"])
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", g["max_scene_cloud_pct"]))
    )
    probs = ee.ImageCollection(g["cloud_prob_collection"]).filterBounds(region).filterDate(start, end)
    joined = ee.Join.saveFirst("cloud_prob").apply(
        primary=sr,
        secondary=probs,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined).map(lambda img: _mask_s2(ee.Image(img), g))


def seasonal_features(cfg: dict, period: str, region: ee.Geometry) -> tuple[ee.Image, ee.Number]:
    """NDVI/NDMI/NBR and mean valid-observation count on the 30 m grid for one period."""
    col = masked_collection(cfg, period, region)
    native = ee.Image(col.first()).select("red").projection()  # 10 m UTM grid
    target = target_projection(cfg)

    median = col.median().setDefaultProjection(native)
    count = col.select("red").count().setDefaultProjection(native)

    refl = median.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(target)
    obs = count.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(target)

    sfx = "before" if period == "before" else "after"
    img = ee.Image.cat([
        refl.normalizedDifference(["nir", "red"]).rename(f"ndvi_{sfx}"),
        refl.normalizedDifference(["nir", "swir1"]).rename(f"ndmi_{sfx}"),
        refl.normalizedDifference(["nir", "swir2"]).rename(f"nbr_{sfx}"),
        obs.rename(f"obs_{sfx}"),
    ])
    return img, col.size()


# --------------------------------------------------------------------------- labels

def label_layers(cfg: dict) -> ee.Image:
    """Per-cell Hansen class fractions plus the binary 'label' band (masked if ineligible)."""
    g = cfg["gee"]
    hansen = ee.Image(g["hansen_asset"])
    lossyear = hansen.select("lossyear")
    target_year = g["loss_year"]

    parts = ee.Image.cat([
        lossyear.eq(target_year).rename("loss_frac"),
        lossyear.eq(0).rename("stable_frac"),
        lossyear.gt(0).And(lossyear.neq(target_year)).rename("other_frac"),
        hansen.select("datamask").eq(1).rename("land_frac"),
        hansen.select("treecover2000").rename("treecover"),
    ]).toFloat()
    frac = parts.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(target_projection(cfg))

    # Area-weighted means of all-ones can land a hair under 1.0, hence 0.999.
    forest = frac.select("treecover").gte(g["treecover_threshold"]).And(frac.select("land_frac").gte(0.999))
    is_loss = frac.select("loss_frac").gte(g["loss_fraction_min"]).And(frac.select("other_frac").eq(0))
    is_stable = frac.select("stable_frac").gte(0.999)
    label = is_loss.rename("label").toInt().updateMask(forest.And(is_loss.Or(is_stable)))
    return frac.addBands(label)


# --------------------------------------------------------------------------- grid ids

def coordinate_bands(cfg: dict) -> ee.Image:
    crs = cfg["gee"]["crs"]
    return ee.Image.cat([
        ee.Image.pixelCoordinates(ee.Projection(crs)).rename(["x", "y"]),
        ee.Image.pixelLonLat().rename(["lon", "lat"]),
    ])


def block_index(cfg: dict) -> ee.Image:
    """Numeric block id (brow * 100000 + bcol), used for server-side grouping only."""
    g = cfg["gee"]
    xy = ee.Image.pixelCoordinates(ee.Projection(g["crs"]))
    bcol = xy.select("x").divide(g["block_m"]).floor()
    brow = xy.select("y").divide(g["block_m"]).floor()
    return brow.multiply(100000).add(bcol).toInt64().rename("block")


def _add_ids(cfg: dict):
    g = cfg["gee"]

    def fn(f):
        f = ee.Feature(f)
        x, y = ee.Number(f.get("x")), ee.Number(f.get("y"))
        col = x.divide(g["grid_m"]).floor().int()
        row = y.divide(g["grid_m"]).floor().int()
        bcol = x.divide(g["block_m"]).floor().int()
        brow = y.divide(g["block_m"]).floor().int()
        return f.set({
            "cell_id": row.format("%d").cat("_").cat(col.format("%d")),
            "block_id": brow.format("%d").cat("_").cat(bcol.format("%d")),
        })

    return fn


# --------------------------------------------------------------------------- assembly

def analysis_image(cfg: dict, region: ee.Geometry) -> tuple[ee.Image, dict]:
    """All sampled bands, masked to eligible, quality-filtered forest cells."""
    g = cfg["gee"]
    before, n_before = seasonal_features(cfg, "before", region)
    after, n_after = seasonal_features(cfg, "after", region)
    labels = label_layers(cfg)
    quality = before.select("obs_before").gte(g["min_valid_obs"]).And(
        after.select("obs_after").gte(g["min_valid_obs"])
    )
    img = ee.Image.cat([before, after, labels, coordinate_bands(cfg)])
    img = img.updateMask(labels.select("label").mask().And(quality))
    return img, {"n_images_before": n_before, "n_images_after": n_after}


def stratified_sample(cfg: dict, img: ee.Image, region: ee.Geometry) -> ee.FeatureCollection:
    g, s = cfg["gee"], cfg["gee"]["sample"]
    sample = img.stratifiedSample(
        numPoints=s["n_stable"],
        classBand="label",
        region=region,
        scale=g["grid_m"],
        projection=ee.Projection(g["crs"]),
        seed=cfg["project"]["seed"],
        classValues=[0, 1],
        classPoints=[s["n_stable"], s["n_loss"]],
        dropNulls=True,
        tileScale=s["tile_scale"],
        geometries=False,
    )
    return sample.map(_add_ids(cfg))


def export_selectors() -> tuple[list[str], list[str]]:
    return list(FEATURE_CSV_COLS), list(LABEL_CSV_COLS)
