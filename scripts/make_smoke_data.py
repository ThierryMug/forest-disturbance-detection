"""Write SYNTHETIC features.csv / labels.csv with the Earth Engine export schema.

For testing the pipeline end to end before (or without) Earth Engine access only.
Values are invented; never report results computed from these files. A few
duplicates, conflicts, missing values and out-of-range values are injected so the
validation steps have something to catch.

    uv run python scripts/make_smoke_data.py --config config/smoke.yaml
"""
import numpy as np
import pandas as pd

from fdd.config import cli_config, get_path


def add_args(p):
    p.add_argument("--n", type=int, default=12000)


cfg, args = cli_config(__doc__, add_args)
rng = np.random.default_rng(0)
grid, block = cfg["gee"]["grid_m"], cfg["gee"]["block_m"]
x0, y0, side = 390000, 6660000, cfg["gee"]["aoi"]["size_m"]

# Unique random cells in a 30 km square.
ncell = side // grid
flat = rng.choice(ncell * ncell, size=args.n, replace=False)
col = x0 // grid + flat % ncell
row = y0 // grid + flat // ncell
x, y = col * grid + grid / 2, row * grid + grid / 2

# Spatially clustered loss: clear-cut-like patches.
centres = rng.uniform([x0, y0], [x0 + side, y0 + side], size=(140, 2))
radii = rng.uniform(120, 450, size=140)
d = np.sqrt((x[:, None] - centres[:, 0]) ** 2 + (y[:, None] - centres[:, 1]) ** 2)
loss = (d < radii).any(axis=1) & (rng.random(args.n) < 0.9)

before = np.column_stack([rng.normal(0.78, 0.05, args.n), rng.normal(0.30, 0.05, args.n), rng.normal(0.55, 0.06, args.n)])
delta = rng.multivariate_normal([0, 0, 0], [[9e-4, 6e-4, 7e-4], [6e-4, 9e-4, 7e-4], [7e-4, 7e-4, 1.2e-3]], args.n)
strong = loss & (rng.random(args.n) < 0.7)
weak = loss & ~strong  # partial harvest: small NDVI drop, clearer moisture/burn-ratio drop
delta[strong] += np.column_stack([rng.normal(-0.25, 0.10, strong.sum()), rng.normal(-0.25, 0.08, strong.sum()), rng.normal(-0.40, 0.12, strong.sum())])
delta[weak] += np.column_stack([rng.normal(-0.04, 0.03, weak.sum()), rng.normal(-0.14, 0.05, weak.sum()), rng.normal(-0.18, 0.06, weak.sum())])
after = np.clip(before + delta, -1, 1)
before = np.clip(before, -1, 1)

feat = pd.DataFrame({
    "cell_id": [f"{r}_{c}" for r, c in zip(row, col)],
    "block_id": [f"{int(yy // block)}_{int(xx // block)}" for xx, yy in zip(x, y)],
    "x": x, "y": y,
    "lon": 15 + (x - 500000) / 55600, "lat": 60 + (y - 6651000) / 111200,
    "ndvi_before": before[:, 0], "ndmi_before": before[:, 1], "nbr_before": before[:, 2],
    "ndvi_after": after[:, 0], "ndmi_after": after[:, 1], "nbr_after": after[:, 2],
    "obs_before": np.clip(rng.normal(9, 3, args.n), 0, None), "obs_after": np.clip(rng.normal(8, 3, args.n), 0, None),
})
lab = pd.DataFrame({
    "cell_id": feat["cell_id"],
    "label": loss.astype(int),
    "loss_frac": np.where(loss, rng.uniform(0.5, 1, args.n), 0.0),
    "stable_frac": np.where(loss, 0.0, 1.0),
    "other_frac": 0.0, "land_frac": 1.0,
    "treecover": rng.uniform(30, 100, args.n),
})
lab["stable_frac"] = np.where(loss, 1 - lab["loss_frac"], 1.0)

# Injected problems.
feat.loc[rng.choice(args.n, 10, replace=False), "ndmi_after"] = np.nan
feat.loc[rng.choice(args.n, 3, replace=False), "ndvi_before"] = 1.7
conflict = feat.sample(5, random_state=1).assign(nbr_after=0.1)
feat = pd.concat([feat, feat.sample(20, random_state=2), conflict], ignore_index=True)
lab = pd.concat([lab, lab.sample(20, random_state=3)], ignore_index=True)
lab = lab.drop(lab.index[rng.choice(args.n, 15, replace=False)])

out = get_path(cfg, "raw_dir")
feat.to_csv(out / "features.csv", index=False)
lab.to_csv(out / "labels.csv", index=False)
print(f"SYNTHETIC smoke data: {len(feat):,} feature rows, {len(lab):,} label rows, loss share {loss.mean():.3f} -> {out}")
