import numpy as np
import pandas as pd

from fdd.splits import buffer_mask, make_splits


def _cfg(buffer_m=60):
    return {
        "project": {"seed": 1},
        "gee": {"block_m": 3000},
        "split": {"fractions": {"train": 0.6, "val": 0.2, "test": 0.2}, "buffer_m": buffer_m,
                  "min_loss_per_split": 5, "min_stable_per_split": 20, "n_trials": 30},
    }


def _cells(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 30000, n)
    y = rng.uniform(0, 30000, n)
    return pd.DataFrame({"cell_id": [f"c{i}" for i in range(n)], "x": x, "y": y,
                         "label": (rng.random(n) < 0.15).astype(int)})


def test_blocks_are_not_shared_between_splits():
    df = _cells()
    out, report = make_splits(df, _cfg())
    blocks = (np.floor(df["x"] / 3000).astype(int).astype(str) + "_" + np.floor(df["y"] / 3000).astype(int).astype(str))
    per_block = out.assign(block=blocks.values).groupby("block")["split"].nunique()
    assert (per_block == 1).all()
    assert sum(d["cells"] for d in report["per_split"].values()) + report["cells_buffer_dropped"] == len(df)


def test_kept_cells_are_at_least_buffer_from_other_splits():
    df = _cells()
    out, _ = make_splits(df, _cfg(buffer_m=90))
    kept = df.assign(split=out["split"].values)[~out["buffer_dropped"].values]
    xy = kept[["x", "y"]].to_numpy()
    for s in ("train", "val", "test"):
        a, b = xy[kept["split"].values == s], xy[kept["split"].values != s]
        # each kept cell is >= buffer from any other-split block, so across an edge or a
        # corner two kept cells of different splits are at least buffer * sqrt(2) apart
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min()
        assert d >= 90 * np.sqrt(2) - 1e-6


def test_buffer_mask_edge_and_corner():
    # one cell near the right edge of block (0,0); neighbour (1,0) has another split
    x, y = np.array([2970.0, 1500.0]), np.array([1500.0, 1500.0])
    bcol, brow = np.array([0, 0]), np.array([0, 0])
    drop = buffer_mask(x, y, bcol, brow, np.array([0, 0]), {(0, 0): 0, (1, 0): 2}, 3000, 60)
    assert drop.tolist() == [True, False]
