"""Column names shared by the extraction, preparation and modeling steps."""

INDEXES = ("ndvi", "ndmi", "nbr")
BEFORE = [f"{i}_before" for i in INDEXES]
AFTER = [f"{i}_after" for i in INDEXES]
CHANGE = [f"d_{i}" for i in INDEXES]  # after minus before, computed in Spark
OBS = ["obs_before", "obs_after"]

ID_COLS = ["cell_id", "block_id", "x", "y", "lon", "lat"]

# Columns of the two Earth Engine CSV exports, joined by cell_id in Spark.
FEATURE_CSV_COLS = ID_COLS + BEFORE + AFTER + OBS
LABEL_CSV_COLS = ["cell_id", "label", "loss_frac", "stable_frac", "other_frac", "land_frac", "treecover"]

# Autoencoder input configurations (proposal section 5).
FEATURE_SETS = {
    "A": ["d_ndvi"],
    "B": CHANGE,
    "C": BEFORE + AFTER + CHANGE,
}

METHODS = ["baseline", "A", "B", "C"]
