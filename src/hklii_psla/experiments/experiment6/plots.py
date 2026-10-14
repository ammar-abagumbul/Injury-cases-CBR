"""Experiment 6 — result figures (matplotlib, written as PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .modeling import MODEL_LABELS

_COLOURS = {"random_forest": "#2a7f62", "logistic_regression": "#3b6fb6", "majority": "#9a9a9a"}


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def label_distribution(raw_counts: dict[str, int], tier_counts: dict[str, int], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, counts, title in (
        (axes[0], raw_counts, "Human labels (raw)"),
        (axes[1], tier_counts, "Tiers used for training"),
    ):
        bars = ax.bar(list(counts), list(counts.values()), color="#5b7db1")
        ax.bar_label(bars)
        ax.set_title(title)
        ax.set_ylabel("cases")
        ax.tick_params(axis="x", rotation=20)
    return _save(fig, path)


def metrics_comparison(
    summaries: dict[str, dict[str, dict]], metrics: tuple[str, ...], path: Path
) -> Path:
    """``summaries[feature_set][model] = CVResult.summary()``."""
    sets = list(summaries)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        models = list(next(iter(summaries.values())))
        width = 0.8 / len(models)
        for j, model in enumerate(models):
            means = [summaries[s][model][metric]["mean"] for s in sets]
            errs = [summaries[s][model][metric]["std"] for s in sets]
            ax.bar(np.arange(len(sets)) + j * width, means, width, yerr=errs, capsize=3,
                   label=MODEL_LABELS.get(model, model), color=_COLOURS.get(model))
        ax.set_xticks(np.arange(len(sets)) + width * (len(models) - 1) / 2, sets, rotation=15)
        ax.set_title(metric.replace("_", " "))
        ax.axhline(0, color="black", linewidth=0.6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def confusion_matrices(
    matrices: dict[str, np.ndarray], class_names: list[str], path: Path, title: str
) -> Path:
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.4 * len(matrices), 4), squeeze=False)
    for ax, (model, cm) in zip(axes[0], matrices.items()):
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(class_names)), class_names, rotation=20)
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("human label")
        ax.set_title(MODEL_LABELS.get(model, model))
    fig.suptitle(title + " (rows sum to 1 = recall per class)")
    return _save(fig, path)


def importance_bars(importances: dict[str, pd.DataFrame], top_n: int, path: Path, title: str) -> Path:
    fig, axes = plt.subplots(1, len(importances), figsize=(6.2 * len(importances), 0.34 * top_n + 1.8), squeeze=False)
    for ax, (model, df) in zip(axes[0], importances.items()):
        top = df.head(top_n).iloc[::-1]
        ax.barh(top["feature"], top["permutation_mean"], xerr=top["permutation_std"],
                color=_COLOURS.get(model), capsize=2)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("drop in macro-F1 when feature is shuffled")
    fig.suptitle(title)
    return _save(fig, path)


def coefficient_heatmap(coefs: pd.DataFrame, order: list[str], top_n: int, path: Path) -> Path:
    use = coefs.loc[[f for f in order if f in coefs.index][:top_n]]
    lim = float(np.abs(use.values).max()) or 1.0
    fig, ax = plt.subplots(figsize=(1.6 * len(use.columns) + 3, 0.32 * len(use) + 1.6))
    im = ax.imshow(use.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(use.columns)), list(use.columns))
    ax.set_yticks(range(len(use)), list(use.index))
    for i in range(use.shape[0]):
        for j in range(use.shape[1]):
            ax.text(j, i, f"{use.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="coefficient (standardised features)")
    ax.set_title("Logistic regression: which features push towards which tier")
    return _save(fig, path)


def feature_distributions(
    X: pd.DataFrame, tiers: pd.Series, features: list[str], tier_order: list[str], path: Path
) -> Path:
    """Per-tier distribution of the most important features.

    Binary features are shown as the share of cases having them; everything else as boxplots.
    """
    n = len(features)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, feat in zip(axes.flat, features):
        values = X[feat]
        is_binary = set(values.dropna().unique()) <= {0.0, 1.0}
        if is_binary:
            share = [values[tiers == t].mean() for t in tier_order]
            ax.bar(tier_order, share, color="#5b7db1")
            ax.set_ylim(0, 1)
            ax.set_ylabel("share of cases")
        else:
            data = [values[tiers == t].dropna().values for t in tier_order]
            ax.boxplot(data, tick_labels=tier_order, showfliers=True)
            ax.set_ylabel(feat)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return _save(fig, path)
