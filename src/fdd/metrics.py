"""Detection metrics, threshold selection and paired block bootstrap."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing F1 (predict positive when score >= threshold)."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    p, r = precision[:-1], recall[:-1]  # the final PR point has no threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {
        "tp": int((y & pred).sum()),
        "fp": int((~y & pred).sum()),
        "fn": int((y & ~pred).sum()),
        "tn": int((~y & ~pred).sum()),
    }


def f1_at(y: np.ndarray, score: np.ndarray, threshold: float) -> float:
    c = confusion(y, score >= threshold)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return 2 * c["tp"] / denom if denom else 0.0


def detection_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    c = confusion(y, score >= threshold)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": float(average_precision_score(y, score)),
        **c,
    }


Scored = tuple[np.ndarray, float]  # (scores, threshold) for one fitted model/seed


def paired_block_bootstrap(
    y: np.ndarray,
    blocks: np.ndarray,
    method_a: Sequence[Scored],
    method_b: Sequence[Scored],
    metric: Callable[[np.ndarray, np.ndarray, float], float],
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap (b - a) by resampling test blocks with replacement.

    Each method is a list of (score, threshold) per seed; the difference is averaged
    over seeds within each replicate. A single-element list is reused for every seed
    (the deterministic baseline). Resampling blocks, not cells, respects spatial
    dependence within blocks.
    """
    n_seeds = max(len(method_a), len(method_b))
    a = list(method_a) * (n_seeds if len(method_a) == 1 else 1)
    b = list(method_b) * (n_seeds if len(method_b) == 1 else 1)

    def diff(idx):
        yy = y[idx]
        return float(np.mean([metric(yy, sb[idx], tb) - metric(yy, sa[idx], ta) for (sa, ta), (sb, tb) in zip(a, b)]))

    point = diff(np.arange(len(y)))
    uniq = np.unique(blocks)
    members = [np.flatnonzero(blocks == u) for u in uniq]
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():  # need both classes
            continue
        reps.append(diff(idx))
    reps = np.array(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "difference": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "share_positive": float((reps > 0).mean()),
        "n_replicates": int(len(reps)),
        "n_blocks": int(len(uniq)),
    }


def ap_metric(y, score, _threshold):
    return average_precision_score(y, score)


def f1_metric(y, score, threshold):
    return f1_at(y, score, threshold)
