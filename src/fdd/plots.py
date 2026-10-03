"""Static figures for the report (matplotlib, PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

# Fixed categorical order (validated reference palette): colour follows the method.
METHOD_COLORS = {"baseline": "#52514e", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
METHOD_LABELS = {
    "baseline": "NDVI threshold",
    "A": "AE-A: dNDVI",
    "B": "AE-B: dNDVI, dNDMI, dNBR",
    "C": "AE-C: before/after + changes",
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
STABLE = "#c3c2b7"  # comparison class in histograms


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_title(title or "", loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel(xlabel or "", color=MUTED)
    ax.set_ylabel(ylabel or "", color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pr_curves(y, scores: dict, prevalence: float, path: Path):
    """scores: method -> score array (one seed)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=METHOD_COLORS[m], linewidth=2, label=METHOD_LABELS[m])
    ax.axhline(prevalence, color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.99, prevalence, f"sampled prevalence {prevalence:.2f}", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax, "Precision-recall, test blocks", "Recall", "Precision")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    _save(fig, path)


def score_distributions(y, scores: dict, thresholds: dict, path: Path):
    fig, axes = plt.subplots(1, len(scores), figsize=(3.2 * len(scores), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (m, s) in zip(axes, scores.items()):
        lo, hi = np.percentile(s, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax.hist(s[y == 0], bins=bins, density=True, color=STABLE, alpha=0.8, label="no mapped loss")
        ax.hist(s[y == 1], bins=bins, density=True, color=METHOD_COLORS[m], alpha=0.75, label="mapped 2024 loss")
        ax.axvline(thresholds[m], color=TEXT, linewidth=1)
        _style(ax, METHOD_LABELS[m], "score", None)
        ax.title.set_fontsize(9)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, path)


def confusion_grid(conf: dict, path: Path):
    """conf: method -> dict(tp, fp, fn, tn)."""
    fig, axes = plt.subplots(1, len(conf), figsize=(2.8 * len(conf), 2.8))
    axes = np.atleast_1d(axes)
    for ax, (m, c) in zip(axes, conf.items()):
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                    color="white" if v > mat.max() * 0.6 else TEXT)
        ax.set_xticks([0, 1], ["pred. no loss", "pred. loss"], fontsize=7)
        ax.set_yticks([0, 1], ["no loss", "loss"], fontsize=7)
        ax.set_title(METHOD_LABELS[m], fontsize=8, loc="left")
    _save(fig, path)


def outcome_map(df, pred, y, title: str, path: Path):
    """Test cells: detected loss (TP), missed loss (FN), false alarms (FP)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    tn = (~pred) & (y == 0)
    ax.scatter(df["x"][tn] / 1000, df["y"][tn] / 1000, s=1, color=GRID, label="correct no-loss", rasterized=True)
    for mask, color, label in [
        (pred & (y == 1), "#2a78d6", "detected loss (TP)"),
        ((~pred) & (y == 1), "#eb6834", "missed loss (FN)"),
        (pred & (y == 0), "#1baf7a", "false alarm (FP)"),
    ]:
        ax.scatter(df["x"][mask] / 1000, df["y"][mask] / 1000, s=5, color=color, label=f"{label} ({mask.sum():,})", rasterized=True)
    ax.set_aspect("equal")
    _style(ax, title, "Easting (km, SWEREF99 TM)", "Northing (km)")
    ax.legend(frameon=False, fontsize=7, markerscale=2.5, loc="upper left", bbox_to_anchor=(1, 1))
    _save(fig, path)


def split_map(df, path: Path):
    colors = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(6, 6))
    dropped = df["buffer_dropped"].to_numpy()
    ax.scatter(df["x"][dropped] / 1000, df["y"][dropped] / 1000, s=1, color=GRID, label="buffer (dropped)", rasterized=True)
    for s, c in colors.items():
        m = (df["split"].to_numpy() == s) & ~dropped
        ax.scatter(df["x"][m] / 1000, df["y"][m] / 1000, s=1, color=c, label=s, rasterized=True)
    loss = (df["label"].to_numpy() == 1) & ~dropped
    ax.scatter(df["x"][loss] / 1000, df["y"][loss] / 1000, s=3, facecolors="none", edgecolors=TEXT, linewidths=0.3, label="mapped 2024 loss")
    ax.set_aspect("equal")
    _style(ax, "Geographic block splits", "Easting (km, SWEREF99 TM)", "Northing (km)")
    ax.legend(frameon=False, fontsize=7, markerscale=4, loc="upper left", bbox_to_anchor=(1, 1))
    _save(fig, path)


def runtime_vs_rows(table, y_col: str, group_col: str | None, title: str, ylabel: str, path: Path):
    """table: tidy DataFrame with rows, median, min, max per (group)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [(None, table)] if group_col is None else list(table.groupby(group_col, sort=False))
    for g, t in groups:
        color = METHOD_COLORS.get(g, "#2a78d6") if g else "#2a78d6"
        ax.errorbar(t["rows"], t[f"{y_col}_median"],
                    yerr=[t[f"{y_col}_median"] - t[f"{y_col}_min"], t[f"{y_col}_max"] - t[f"{y_col}_median"]],
                    color=color, marker="o", markersize=6, linewidth=2, capsize=3,
                    label=METHOD_LABELS.get(g, g) if g else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, title, "Rows (log)", ylabel)
    if group_col is not None:
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)
