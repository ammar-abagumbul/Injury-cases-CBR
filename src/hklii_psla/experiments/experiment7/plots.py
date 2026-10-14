"""Experiment 7 — figures (matplotlib PNGs)."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

from .analysis import TreeReport

_SIG, _NS = "#c4572f", "#b7bcc4"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _wrap(text: str, width: int = 44) -> str:
    return "\n".join(textwrap.wrap(text, width)) or text


def court_by_tier(table: pd.DataFrame, path: Path) -> Path:
    """Stacked bars: how many cases of each tier come from each court."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bottom = np.zeros(len(table.index))
    for court in table.columns:
        ax.bar(table.index, table[court].to_numpy(), bottom=bottom, label=court)
        bottom += table[court].to_numpy()
    for i, total in enumerate(bottom):
        ax.text(i, total, int(total), ha="center", va="bottom")
    ax.set_ylabel("cases")
    ax.set_title("Which court each tier's cases come from")
    ax.legend(title="court")
    return _save(fig, path)


def tier_profile(
    X: pd.DataFrame, tiers: pd.Series, features: list[str], kinds: dict[str, str],
    tier_order: list[str], explain: Callable[[str], str], path: Path,
) -> Path:
    """Heatmap: each row is a feature, each column a tier. Binary features show the share of
    cases with it, numeric ones the median (among cases that state it). Colour is scaled per row."""
    raw = np.full((len(features), len(tier_order)), np.nan)
    for j, tier in enumerate(tier_order):
        sub = X[(tiers == tier).to_numpy()]
        for i, f in enumerate(features):
            v = sub[f].dropna()
            if len(v):
                raw[i, j] = v.mean() if kinds[f] == "binary" else v.median()
    scaled = np.zeros_like(raw)
    for i in range(len(features)):
        lo, hi = np.nanmin(raw[i]), np.nanmax(raw[i])
        scaled[i] = 0 if hi == lo else (raw[i] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(7.5 + 0.2 * len(tier_order), 0.42 * len(features) + 1.8))
    ax.imshow(scaled, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tier_order)), tier_order)
    ax.set_yticks(range(len(features)), [_wrap(explain(f), 52) for f in features], fontsize=8)
    for i, f in enumerate(features):
        for j in range(len(tier_order)):
            if np.isnan(raw[i, j]):
                label = "—"
            elif kinds[f] == "binary":
                label = f"{raw[i, j]:.0%}"
            else:
                label = f"{raw[i, j]:.3g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if scaled[i, j] > 0.65 else "black")
    ax.set_title("Tier profile: share of cases (flags) or median value (numbers)\ncolour scaled per row")
    return _save(fig, path)


def top_differences(
    tables: dict[str, pd.DataFrame], explain: Callable[[str], str], alpha: float, top_n: int, path: Path
) -> Path | None:
    """For each comparison, the features with the biggest effect (orange = passes FDR)."""
    tables = {k: v for k, v in tables.items() if v is not None and not v.empty}
    if not tables:
        return None
    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 0.4 * top_n + 2), squeeze=False)
    for ax, (label, df) in zip(axes[0], tables.items()):
        top = df.reindex(df["effect"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
        colours = [_SIG if q < alpha else _NS for q in top["q"]]
        ax.barh([_wrap(explain(f), 40) for f in top["feature"]], top["effect"], color=colours)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("effect (risk difference / Cliff's delta)")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Biggest differences; orange = significant after FDR correction (q < {alpha}), grey = exploratory")
    return _save(fig, path)


def decision_tree(report: TreeReport, explain: Callable[[str], str], path: Path) -> Path:
    """The fitted tree with plain-language questions and real (unweighted) case counts."""
    model, t = report.model, report.model.tree_
    fig, ax = plt.subplots(figsize=(17, 7.5))
    nodes = plot_tree(
        model, feature_names=report.feature_names, class_names=report.class_names,
        filled=True, rounded=True, impurity=False, fontsize=8, ax=ax,
    )
    boxes = [a for a in nodes if "samples" in a.get_text()]  # node boxes only, not the True/False edge labels
    for i, art in enumerate(boxes):
        counts = report.node_counts[i] if report.node_counts is not None else None
        lines = [_wrap(report.node_text[i], 34)] if report.node_text and report.node_text[i] else []
        if counts is not None:
            lines.append(f"{int(counts.sum())} cases")
            lines += [f"{c}: {int(k)}" for c, k in zip(report.class_names, counts) if k]
        lines.append(f"leaning {report.class_names[int(np.argmax(t.value[i][0]))]}")
        art.set_text("\n".join(lines))
    ax.set_title("Decision tree: follow 'True' (left) or 'False' (right). Colour = tier the box leans to "
                 "(tiers are weighted equally); numbers are real case counts")
    return _save(fig, path)


def tree_stability(stability: pd.DataFrame, explain: Callable[[str], str], path: Path, top_n: int = 12) -> Path:
    top = stability.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1.8))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["share_of_trees_using"], 0.4, label="used anywhere in the tree", color="#9aa7b8")
    ax.barh(y - 0.2, top["share_as_first_split"], 0.4, label="used as the first split", color=_SIG)
    ax.set_yticks(y, [_wrap(explain(f), 46) for f in top["feature"]], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of cross-validation trees")
    ax.set_title("How stable the tree's choice of features is")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)
