"""Experiment 7 — statistics and decision rules for "what makes a case fall in a tier".

Everything here is deterministic and LLM-free:

* ``compare``           one feature-by-feature comparison of two groups of cases
                        (Fisher exact test for flags, Mann-Whitney for numbers)
* ``trend``             Spearman correlation of each feature with the tier order
* ``court_*``           measures of how much court (High Court vs District Court)
                        is mixed up with tier, and which features merely proxy court
* ``fit_tree`` etc.     a shallow decision tree turned into readable if/then rules,
                        with its cross-validated reliability and split stability
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from hklii_psla.experiments.experiment6.modeling import score

# ---------------------------------------------------------------- feature kinds


def feature_kinds(X: pd.DataFrame) -> dict[str, str]:
    """'binary' (only 0/1 values) or 'numeric'."""
    return {
        c: "binary" if set(X[c].dropna().unique()) <= {0.0, 1.0} else "numeric"
        for c in X.columns
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    return stats.false_discovery_control(np.asarray(p, dtype=float), method="bh") if len(p) else np.asarray([])


# ------------------------------------------------------------------ comparisons


def compare(
    X: pd.DataFrame, kinds: dict[str, str], a: np.ndarray, b: np.ndarray, min_valid: int = 3
) -> pd.DataFrame:
    """Every feature, group A vs group B (``a`` / ``b`` are boolean masks).

    ``effect`` is the risk difference (flags) or Cliff's delta (numbers); both lie in
    [-1, 1] and are positive when the feature is higher in group A.
    """
    rows: list[dict[str, Any]] = []
    for col in X.columns:
        va, vb = X.loc[a, col].dropna().to_numpy(), X.loc[b, col].dropna().to_numpy()
        if len(va) < min_valid or len(vb) < min_valid:
            continue
        row: dict[str, Any] = {
            "feature": col, "kind": kinds[col], "n_a": len(va), "n_b": len(vb),
            "stated_a": len(va) / max(int(a.sum()), 1), "stated_b": len(vb) / max(int(b.sum()), 1),
        }
        if kinds[col] == "binary":
            ka, kb = int(va.sum()), int(vb.sum())
            if ka + kb == 0:
                continue
            table = [[ka, len(va) - ka], [kb, len(vb) - kb]]
            row.update(
                stat_a=ka / len(va), stat_b=kb / len(vb),
                ci_a=wilson(ka, len(va)), ci_b=wilson(kb, len(vb)),
                effect=ka / len(va) - kb / len(vb), effect_type="risk difference",
                odds_ratio=((ka + .5) * (len(vb) - kb + .5)) / ((len(va) - ka + .5) * (kb + .5)),
                p=float(stats.fisher_exact(table)[1]),
            )
        else:
            if np.ptp(np.concatenate([va, vb])) == 0:
                continue
            u = stats.mannwhitneyu(va, vb, alternative="two-sided")
            row.update(
                stat_a=float(np.median(va)), stat_b=float(np.median(vb)),
                mean_a=float(va.mean()), mean_b=float(vb.mean()),
                ci_a=None, ci_b=None,
                effect=float(2 * u.statistic / (len(va) * len(vb)) - 1), effect_type="Cliff's delta",
                odds_ratio=float("nan"), p=float(u.pvalue),
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q"] = bh(df["p"].to_numpy())
    return df


def trend(X: pd.DataFrame, y: np.ndarray, min_valid: int = 10) -> pd.DataFrame:
    """Spearman correlation of each feature with the tier order (positive = rises with severity)."""
    rows = []
    for col in X.columns:
        m = X[col].notna().to_numpy()
        v = X.loc[m, col].to_numpy()
        if m.sum() < min_valid or np.ptp(v) == 0 or np.ptp(y[m]) == 0:
            continue
        rho, p = stats.spearmanr(v, y[m])
        rows.append({"feature": col, "rho": float(rho), "p": float(p), "n": int(m.sum())})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = bh(df["p"].to_numpy())
    return df


# --------------------------------------------------------------- court analysis


def court_distribution(courts: pd.Series, mask: np.ndarray) -> pd.Series:
    return courts[mask].value_counts(normalize=True)


def total_variation(courts: pd.Series, a: np.ndarray, b: np.ndarray) -> float:
    """0 = both groups have the same court mix, 1 = no court in common."""
    pa, pb = court_distribution(courts, a), court_distribution(courts, b)
    idx = pa.index.union(pb.index)
    return float(0.5 * np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())


def court_proxy_auc(X: pd.DataFrame, courts: pd.Series, min_group: int = 5) -> pd.Series:
    """Per-feature AUC for telling the two most common courts apart (0.5 = no information).

    A feature far from 0.5 here says as much about the court as about the injury.
    """
    top = courts.value_counts()
    if len(top) < 2 or top.iloc[1] < min_group:
        return pd.Series(dtype=float)
    c1, c2 = top.index[:2]
    m = courts.isin([c1, c2]).to_numpy()
    yc = (courts[m] == c1).astype(int).to_numpy()
    out = {}
    for col in X.columns:
        v = X.loc[m, col]
        ok = v.notna().to_numpy()
        if ok.sum() >= 10 and len(set(yc[ok])) == 2 and v[ok].nunique() > 1:
            out[col] = float(roc_auc_score(yc[ok], v[ok]))
    return pd.Series(out, name=f"court_auc[{c1} vs {c2}]")


# -------------------------------------------------------------------- the tree


@dataclass
class TreeReport:
    rules: pd.DataFrame
    cv: dict[str, Any]
    stability: pd.DataFrame
    model: DecisionTreeClassifier
    feature_names: list[str]
    class_names: list[str]
    node_counts: np.ndarray | None = None  # real (unweighted) cases per class at every node
    node_text: list[str] = field(default_factory=list)  # plain-language split question per node
    notes: list[str] = field(default_factory=list)


def _new_tree(depth: int, leaf: int, seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=leaf, class_weight="balanced", random_state=seed
    )


def _condition(
    model: DecisionTreeClassifier, node: int, went_left: bool, names: list[str], has_nan: set[str],
    label: Callable[[str], str] = lambda f: f,
) -> str:
    t = model.tree_
    raw = names[t.feature[node]]
    feat = label(raw)
    thr = float(t.threshold[node])
    nan_left = bool(t.missing_go_to_left[node]) and raw in has_nan
    nan_right = (not bool(t.missing_go_to_left[node])) and raw in has_nan
    if not np.isfinite(thr):  # sklearn's "has a value / is missing" split
        return f"[{feat}] stated" if went_left else f"[{feat}] not stated"
    binary = abs(thr - 0.5) < 1e-9
    if binary:
        if went_left:
            return f"NOT [{feat}]" + (" (or not stated)" if nan_left else "")
        return f"[{feat}]" + (" (or not stated)" if nan_right else "")
    op = f"[{feat}] <= {thr:.3g}" if went_left else f"[{feat}] > {thr:.3g}"
    if (went_left and nan_left) or (not went_left and nan_right):
        op += " (or not stated)"
    return op


def extract_rules(
    model: DecisionTreeClassifier, X: pd.DataFrame, y: np.ndarray, class_names: list[str],
    label: Callable[[str], str] = lambda f: f,
) -> pd.DataFrame:
    """One row per leaf: the if/then path, predicted tier and the real case counts."""
    names = list(X.columns)
    has_nan = {c for c in names if X[c].isna().any()}
    t = model.tree_
    leaf_of = model.apply(X)
    rows = []

    def walk(node: int, path: list[str]) -> None:
        if t.children_left[node] == -1:
            ids = leaf_of == node
            counts = np.bincount(y[ids], minlength=len(class_names))
            pred = int(np.argmax(t.value[node][0]))
            rows.append({
                "rule": " AND ".join(path) if path else "(always)",
                "predicts": class_names[pred],
                "n_cases": int(ids.sum()),
                **{f"n_{c}": int(k) for c, k in zip(class_names, counts)},
                "precision": float(counts[pred] / counts.sum()) if counts.sum() else float("nan"),
                "depth": len(path),
            })
            return
        walk(t.children_left[node], [*path, _condition(model, node, True, names, has_nan, label)])
        walk(t.children_right[node], [*path, _condition(model, node, False, names, has_nan, label)])

    walk(0, [])
    return pd.DataFrame(rows).sort_values(["predicts", "n_cases"], ascending=[True, False]).reset_index(drop=True)


def fit_tree(
    X: pd.DataFrame, y: np.ndarray, class_names: list[str],
    depth: int, leaf: int, n_splits: int, n_repeats: int, seed: int,
    label: Callable[[str], str] = lambda f: f,
) -> TreeReport:
    """Fit on all cases for the rules; cross-validate the same tree to say how far to trust it."""
    model = _new_tree(depth, leaf, seed).fit(X, y)
    rules = extract_rules(model, X, y, class_names)
    rules.insert(0, "rule_plain", extract_rules(model, X, y, class_names, label)["rule"])

    smallest = int(np.bincount(y).min())
    folds = min(n_splits, smallest)
    notes = []
    if folds < n_splits:
        notes.append(f"Rarest tier has {smallest} cases: cross-validation used {folds} folds instead of {n_splits}.")
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=n_repeats, random_state=seed)

    fold_scores, base_scores = [], []
    used: Counter[str] = Counter()
    root: Counter[str] = Counter()
    n_fits = 0
    for train, test in splitter.split(X, y):
        m = _new_tree(depth, leaf, seed).fit(X.iloc[train], y[train])
        fold_scores.append(score(y[test], m.predict(X.iloc[test]), len(class_names)))
        majority = np.bincount(y[train]).argmax()
        base_scores.append(score(y[test], np.full(len(test), majority), len(class_names)))
        feats = m.tree_.feature[m.tree_.feature >= 0]
        used.update({X.columns[f] for f in feats})
        if len(feats):
            root[X.columns[m.tree_.feature[0]]] += 1
        n_fits += 1

    def mean_std(rows: list[dict], key: str) -> dict[str, float]:
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    cv = {
        "folds": folds, "repeats": n_repeats, "n_fits": n_fits,
        "tree": {k: mean_std(fold_scores, k) for k in ("macro_f1", "qwk", "balanced_accuracy", "accuracy")},
        "majority": {k: mean_std(base_scores, k) for k in ("macro_f1", "qwk", "balanced_accuracy", "accuracy")},
        "recall_tree": np.mean([r["recall"] for r in fold_scores], axis=0).tolist(),
    }
    stability = pd.DataFrame({
        "feature": sorted(set(used) | set(root)),
    })
    stability["share_of_trees_using"] = stability["feature"].map(lambda f: used[f] / n_fits)
    stability["share_as_first_split"] = stability["feature"].map(lambda f: root[f] / n_fits)
    stability = stability.sort_values(["share_as_first_split", "share_of_trees_using"], ascending=False).reset_index(drop=True)

    names = list(X.columns)
    has_nan = {c for c in names if X[c].isna().any()}
    node_counts = np.asarray(model.decision_path(X).T @ np.eye(len(class_names))[y])
    node_text = [
        "" if model.tree_.children_left[i] == -1
        else _condition(model, i, True, names, has_nan, label).replace("[", "").replace("]", "") + "?"
        for i in range(model.tree_.node_count)
    ]
    return TreeReport(rules, cv, stability, model, names, class_names, node_counts, node_text, notes)


# ------------------------------------------------------------------- bundles


@dataclass
class ComparisonResult:
    stratum: str
    label: str
    name_a: str
    name_b: str
    n_a: int
    n_b: int
    court_mix_distance: float  # 0 = same court mix in both groups, 1 = disjoint
    table: pd.DataFrame


@dataclass
class Analysis:
    tier_names: list[str]
    tier_counts: dict[str, int]
    court_tier: pd.DataFrame  # rows = tiers, columns = courts
    comparisons: list[ComparisonResult]
    trends: dict[str, pd.DataFrame]
    court_auc: pd.Series
    trees: dict[str, TreeReport]
    strata_sizes: dict[str, int]
    warnings: list[str]
    alpha: float
    top_n: int
    primary: str
    kinds: dict[str, str] = field(default_factory=dict)
