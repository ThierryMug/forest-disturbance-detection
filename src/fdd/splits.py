"""Geographic block splitting with a border buffer.

Whole blocks go to train/val/test, so adjacent cells stay together. Cells whose
centre lies within buffer_m of a neighbouring block assigned to a *different*
split are dropped from analysis. Many random block assignments are scored and the
one closest to the target fractions (for all cells and for loss cells), that also
meets the per-split minimums, is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def block_coords(df: pd.DataFrame, block_m: float) -> tuple[np.ndarray, np.ndarray]:
    return np.floor(df["x"].to_numpy() / block_m).astype(np.int64), np.floor(df["y"].to_numpy() / block_m).astype(np.int64)


def assign_blocks(block_sizes: np.ndarray, fractions: dict, rng: np.random.Generator) -> np.ndarray:
    """Random order of blocks, cut by cumulative share of cells at each block's midpoint."""
    order = rng.permutation(len(block_sizes))
    sizes = block_sizes[order]
    mid = (np.cumsum(sizes) - sizes / 2) / sizes.sum()
    cut1 = fractions["train"]
    cut2 = fractions["train"] + fractions["val"]
    labels = np.where(mid < cut1, 0, np.where(mid < cut2, 1, 2))
    out = np.empty(len(block_sizes), dtype=np.int8)
    out[order] = labels
    return out


def buffer_mask(x, y, bcol, brow, cell_split, block_split: dict, block_m: float, buffer_m: float) -> np.ndarray:
    """True for cells within buffer_m of an (8-)neighbouring block of another split."""
    drop = np.zeros(len(x), dtype=bool)
    if buffer_m <= 0:
        return drop
    dx_left = x - bcol * block_m
    dx_right = (bcol + 1) * block_m - x
    dy_down = y - brow * block_m
    dy_up = (brow + 1) * block_m - y
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nb = np.array([block_split.get((c + di, r + dj), -1) for c, r in zip(bcol, brow)])
            differs = (nb >= 0) & (nb != cell_split)
            ddx = dx_left if di == -1 else dx_right if di == 1 else 0.0
            ddy = dy_down if dj == -1 else dy_up if dj == 1 else 0.0
            dist = np.sqrt(np.square(ddx) + np.square(ddy))
            drop |= differs & (dist < buffer_m)
    return drop


def make_splits(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return (assignment per cell_id, report). Requires columns cell_id, x, y, label."""
    g, sc = cfg["gee"], cfg["split"]
    block_m, buffer_m = g["block_m"], sc["buffer_m"]
    fractions = sc["fractions"]
    target = np.array([fractions[s] for s in SPLITS])

    x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
    label = df["label"].to_numpy(int)
    bcol, brow = block_coords(df, block_m)
    keys = list(zip(bcol.tolist(), brow.tolist()))
    uniq = sorted(set(keys))
    index = {k: i for i, k in enumerate(uniq)}
    cell_block = np.array([index[k] for k in keys])
    block_sizes = np.bincount(cell_block, minlength=len(uniq))

    rng = np.random.default_rng(cfg["project"]["seed"])
    best = None
    for trial in range(sc["n_trials"]):
        bsplit = assign_blocks(block_sizes, fractions, rng)
        cell_split = bsplit[cell_block]
        drop = buffer_mask(x, y, bcol, brow, cell_split, {k: int(bsplit[i]) for k, i in index.items()}, block_m, buffer_m)
        kept_split, kept_label = cell_split[~drop], label[~drop]
        n_loss = np.array([(kept_label[kept_split == s] == 1).sum() for s in range(3)])
        n_stable = np.array([(kept_label[kept_split == s] == 0).sum() for s in range(3)])
        feasible = (n_loss >= sc["min_loss_per_split"]).all() and (n_stable >= sc["min_stable_per_split"]).all()
        n_all = n_loss + n_stable
        cost = np.abs(n_all / n_all.sum() - target).sum() + np.abs(n_loss / max(n_loss.sum(), 1) - target).sum()
        if best is None or (feasible, -cost) > (best["feasible"], -best["cost"]):
            best = {"trial": trial, "feasible": feasible, "cost": float(cost), "cell_split": cell_split, "drop": drop}

    if not best["feasible"]:
        raise RuntimeError(
            "No block assignment met the per-split minimums; lower split.min_* or block_m, "
            "or extract more loss cells. Record the change in docs/project_log.md."
        )

    names = np.array(SPLITS)
    out = pd.DataFrame({
        "cell_id": df["cell_id"].to_numpy(),
        "split": names[best["cell_split"]],
        "buffer_dropped": best["drop"],
    })
    kept = out[~out["buffer_dropped"]].assign(label=label[~best["drop"]])
    report = {
        "n_blocks": len(uniq),
        "chosen_trial": best["trial"],
        "cost": best["cost"],
        "buffer_m": buffer_m,
        "cells_buffer_dropped": int(best["drop"].sum()),
        "per_split": {
            s: {
                "blocks": int(len(np.unique(cell_block[best["cell_split"] == i]))),
                "cells": int((kept["split"] == s).sum()),
                "loss": int(((kept["split"] == s) & (kept["label"] == 1)).sum()),
                "stable": int(((kept["split"] == s) & (kept["label"] == 0)).sum()),
            }
            for i, s in enumerate(SPLITS)
        },
    }
    for s in SPLITS:
        d = report["per_split"][s]
        d["loss_prevalence"] = d["loss"] / d["cells"] if d["cells"] else float("nan")
    return out, report
