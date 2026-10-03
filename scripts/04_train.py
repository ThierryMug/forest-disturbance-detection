"""Fit the NDVI baseline threshold and autoencoders A/B/C (validation data only for choices).

Protocol per feature set:
  1. scaler + autoencoder fitted on stable TRAIN cells only;
  2. early stopping on stable VALIDATION reconstruction loss;
  3. predeclared tuning grid, best configuration chosen by validation average precision
     (first model seed);
  4. chosen configuration refitted for every model seed; each model's decision threshold
     maximizes validation F1.
Test cells are not touched here.
"""
import itertools

import pandas as pd
from sklearn.metrics import average_precision_score

from fdd.config import cli_config, get_path, save_json, set_seeds
from fdd.metrics import best_f1_threshold
from fdd.models import Scaler, ae_params, baseline_score, load_dataset, reconstruction_error, train_autoencoder
from fdd.schema import FEATURE_SETS


def add_args(p):
    p.add_argument("--sets", nargs="+", default=list(FEATURE_SETS), help="feature sets to train")


cfg, args = cli_config(__doc__, add_args)
set_seeds(cfg["project"]["seed"])
models_dir, res = get_path(cfg, "models_dir"), get_path(cfg, "results_dir")
seeds = cfg["project"]["model_seeds"]

df = load_dataset(cfg)
train, val = df[df["split"] == "train"], df[df["split"] == "val"]
train_stable, val_stable = train[train["label"] == 0], val[val["label"] == 0]
y_val = val["label"].to_numpy()
print(f"train stable={len(train_stable):,} | val={len(val):,} (loss={y_val.sum():,})")

# Baseline: NDVI decrease, threshold from validation F1.
thr, f1 = best_f1_threshold(y_val, baseline_score(val))
save_json({"method": "baseline", "score": "-d_ndvi", "threshold": thr, "val_f1": f1,
           "val_ap": float(average_precision_score(y_val, baseline_score(val)))}, models_dir / "baseline.json")
print(f"baseline: threshold(-dNDVI)={thr:.4f} val F1={f1:.3f}")

grid = cfg["autoencoder"]["tuning_grid"]
grid_keys = list(grid)
tuning_rows, training_rows = [], []

for fs in args.sets:
    cols = FEATURE_SETS[fs]
    scaler = Scaler.fit(train_stable, cols)
    x_tr, x_vs, x_val = scaler.transform(train_stable), scaler.transform(val_stable), scaler.transform(val)
    out_dir = models_dir / fs
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(out_dir / "scaler.json")

    # 1) tuning on the first seed
    best = None
    for values in itertools.product(*(grid[k] for k in grid_keys)):
        params = ae_params(cfg, **dict(zip(grid_keys, values)))
        model, info = train_autoencoder(x_tr, x_vs, params, seeds[0])
        ap = float(average_precision_score(y_val, reconstruction_error(model, x_val)))
        tuning_rows.append({"set": fs, **dict(zip(grid_keys, values)), "val_ap": ap, **info})
        print(f"[{fs}] tune {dict(zip(grid_keys, values))} val AP={ap:.3f} epochs={info['epochs_completed']}")
        if best is None or ap > best[0]:
            best = (ap, params, model, info)
    _, params, best_model, best_info = best
    save_json(params, out_dir / "params.json")

    # 2) final models per seed (the tuning fit is reused for the first seed)
    for seed in seeds:
        if seed == seeds[0]:
            model, info = best_model, best_info
        else:
            model, info = train_autoencoder(x_tr, x_vs, params, seed)
        score_val = reconstruction_error(model, x_val)
        thr, f1 = best_f1_threshold(y_val, score_val)
        model.save(out_dir / f"seed{seed}.keras")
        meta = {"set": fs, "seed": seed, "features": cols, "threshold": thr, "val_f1": f1,
                "val_ap": float(average_precision_score(y_val, score_val)), **info}
        save_json(meta, out_dir / f"seed{seed}.json")
        training_rows.append({k: v for k, v in meta.items() if k != "features"})
        print(f"[{fs}] seed {seed}: val AP={meta['val_ap']:.3f} F1={f1:.3f} epochs={info['epochs_completed']} "
              f"({info['train_seconds']:.1f} s)")

if tuning_rows:
    pd.DataFrame(tuning_rows).to_csv(res / "tuning.csv", index=False)
    pd.DataFrame(training_rows).to_csv(res / "training.csv", index=False)
print(f"Saved models to {models_dir}")
