"""Held-out evaluation on test blocks: metrics, comparisons, figures.

Outputs (results_dir):
  metrics_test.csv        one row per method x seed
  metrics_summary.csv     mean / sd / min / max over seeds
  comparisons.csv         paired block-bootstrap differences in AP and F1
  figures/*.png           PR curves, score distributions, confusion matrices, outcome maps
"""
import keras
import numpy as np
import pandas as pd

from fdd import plots
from fdd.config import cli_config, get_path, load_json, save_json
from fdd.metrics import ap_metric, detection_metrics, f1_metric, paired_block_bootstrap
from fdd.models import Scaler, baseline_score, load_dataset, reconstruction_error
from fdd.schema import FEATURE_SETS

COMPARISONS = [("baseline", "A"), ("A", "B"), ("B", "C"), ("baseline", "C")]

cfg, _ = cli_config(__doc__)
models_dir, res = get_path(cfg, "models_dir"), get_path(cfg, "results_dir")
fig_dir = res / "figures"
seeds = cfg["project"]["model_seeds"]

df = load_dataset(cfg)
test = df[df["split"] == "test"].reset_index(drop=True)
y = test["label"].to_numpy()
blocks = test["block_id"].to_numpy()
print(f"test cells={len(test):,} loss={y.sum():,} prevalence={y.mean():.3f} blocks={len(np.unique(blocks))}")

# scored[method] = list of (score, threshold), one per seed (baseline: one entry)
scored = {"baseline": [(baseline_score(test), load_json(models_dir / "baseline.json")["threshold"])]}
for fs in FEATURE_SETS:
    if not (models_dir / fs).exists():
        continue
    scaler = Scaler.load(models_dir / fs / "scaler.json")
    x = scaler.transform(test)
    scored[fs] = []
    for seed in seeds:
        model = keras.models.load_model(models_dir / fs / f"seed{seed}.keras")
        thr = load_json(models_dir / fs / f"seed{seed}.json")["threshold"]
        scored[fs].append((reconstruction_error(model, x), thr))

rows = []
for m, runs in scored.items():
    for i, (s, thr) in enumerate(runs):
        rows.append({"method": m, "seed": None if m == "baseline" else seeds[i], **detection_metrics(y, s, thr)})
metrics = pd.DataFrame(rows)
metrics.to_csv(res / "metrics_test.csv", index=False)

summary = (
    metrics.groupby("method", sort=False)[["precision", "recall", "f1", "average_precision"]]
    .agg(["mean", "std", "min", "max"])
)
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary.insert(0, "n_seeds", metrics.groupby("method", sort=False).size())
summary.to_csv(res / "metrics_summary.csv")

comparisons = []
for a, b in COMPARISONS:
    if a not in scored or b not in scored:
        continue
    for name, fn in [("average_precision", ap_metric), ("f1", f1_metric)]:
        r = paired_block_bootstrap(y, blocks, scored[a], scored[b], fn, cfg["evaluation"]["n_bootstrap"], cfg["project"]["seed"])
        comparisons.append({"comparison": f"{b} - {a}", "metric": name, **r})
comparisons = pd.DataFrame(comparisons)
comparisons.to_csv(res / "comparisons.csv", index=False)

# Figures use the first seed of each autoencoder.
first = {m: runs[0][0] for m, runs in scored.items()}
thr_first = {m: runs[0][1] for m, runs in scored.items()}
plots.pr_curves(y, first, float(y.mean()), fig_dir / "pr_curves.png")
plots.score_distributions(y, first, thr_first, fig_dir / "score_distributions.png")
plots.confusion_grid({m: metrics[metrics["method"] == m].iloc[0][["tp", "fp", "fn", "tn"]].to_dict() for m in scored},
                     fig_dir / "confusion_matrices.png")
for m in scored:
    plots.outcome_map(test, first[m] >= thr_first[m], y, f"Test outcomes: {plots.METHOD_LABELS[m]}", fig_dir / f"map_{m}.png")

save_json({"n_test": len(test), "n_test_loss": int(y.sum()), "prevalence": float(y.mean()),
           "n_test_blocks": int(len(np.unique(blocks))), "seeds": seeds}, res / "evaluation_meta.json")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
print(summary[["n_seeds", "precision_mean", "recall_mean", "f1_mean", "f1_std", "average_precision_mean", "average_precision_std"]].round(3))
if len(comparisons):
    print(comparisons[["comparison", "metric", "difference", "ci95_low", "ci95_high"]].round(3).to_string(index=False))
print(f"Saved results to {res}")
