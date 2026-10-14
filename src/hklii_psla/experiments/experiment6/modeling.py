"""Experiment 6 — models, cross-validation, metrics and feature importance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import LOG_COLUMNS

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "majority": "Majority class",
}
METRICS = ("macro_f1", "qwk", "balanced_accuracy", "accuracy")


def build_pipeline(
    name: str, columns: list[str], params: dict[str, Any], seed: int
) -> Pipeline:
    """Unfitted pipeline for ``name`` (imputation and scaling happen inside the CV fold)."""
    if name == "random_forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=-1.0, keep_empty_features=True)),
            ("model", RandomForestClassifier(
                **{"n_estimators": 500, "min_samples_leaf": 2, "class_weight": "balanced_subsample",
                   "n_jobs": -1, **params, "random_state": seed},
            )),
        ])

    if name == "logistic_regression":
        log_cols = [c for c in columns if c in LOG_COLUMNS]
        other = [c for c in columns if c not in LOG_COLUMNS]
        logged = Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        plain = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        transform = ColumnTransformer(
            [("log", logged, log_cols), ("plain", plain, other)],
            verbose_feature_names_out=False,
        )
        return Pipeline([
            ("prep", transform),
            ("model", LogisticRegression(
                **{"C": 1.0, "class_weight": "balanced", "max_iter": 5000, **params}
            )),
        ])

    if name == "majority":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ])

    raise ValueError(f"Unknown model {name!r}")


def score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, Any]:
    labels = list(range(n_classes))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "recall": [float(r) for r in recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )],
    }


@dataclass
class CVResult:
    model: str
    folds: list[dict[str, Any]] = field(default_factory=list)
    oof: list[dict[str, int]] = field(default_factory=list)  # sample, repeat, y_true, y_pred
    confusion: np.ndarray | None = None  # summed over all test folds
    n_repeats: int = 1

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric in METRICS:
            vals = np.array([f[metric] for f in self.folds])
            out[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        recalls = np.array([f["recall"] for f in self.folds])
        out["recall"] = {
            "mean": recalls.mean(axis=0).tolist(),
            "std": (recalls.std(axis=0, ddof=1) if len(recalls) > 1 else np.zeros(recalls.shape[1])).tolist(),
        }
        return out

    def normalised_confusion(self) -> np.ndarray:
        assert self.confusion is not None
        rows = self.confusion.sum(axis=1, keepdims=True)
        return np.divide(self.confusion, rows, out=np.zeros_like(self.confusion, dtype=float), where=rows > 0)


def make_splitter(y: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> tuple[RepeatedStratifiedKFold, int]:
    """Stratified repeated K-fold; ``n_splits`` shrinks if the rarest class is too small."""
    smallest = int(np.bincount(y).min())
    if smallest < 2:
        raise ValueError(
            f"A class has only {smallest} labelled case(s); need at least 2 per class for "
            "stratified cross-validation. Extract more cases or merge tiers in the config."
        )
    n_splits = min(n_splits, smallest)
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed), n_splits


def cross_validate(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
    splitter: RepeatedStratifiedKFold,
    params: dict[str, Any],
    seed: int,
) -> CVResult:
    result = CVResult(model=name, n_repeats=splitter.n_repeats)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    template = build_pipeline(name, list(X.columns), params, seed)
    n_splits = splitter.cvargs["n_splits"]

    for i, (train, test) in enumerate(splitter.split(X, y)):
        pipe = clone(template).fit(X.iloc[train], y[train])
        pred = pipe.predict(X.iloc[test])
        result.folds.append({
            "repeat": i // n_splits, "fold": i % n_splits,
            "n_train": len(train), "n_test": len(test), **score(y[test], pred, n_classes),
        })
        confusion += confusion_matrix(y[test], pred, labels=list(range(n_classes)))
        result.oof.extend(
            {"sample": int(s), "repeat": i // n_splits, "y_true": int(t), "y_pred": int(p)}
            for s, t, p in zip(test, y[test], pred)
        )
    result.confusion = confusion
    return result


def compare_models(a: CVResult, b: CVResult, metric: str, n_splits: int) -> dict[str, float]:
    """Paired comparison on identical folds (a minus b).

    Uses the Nadeau-Bengio corrected resampled t-test, because folds from repeated
    CV share training data and the naive paired t-test is overconfident.
    """
    diff = np.array([fa[metric] - fb[metric] for fa, fb in zip(a.folds, b.folds)])
    k = len(diff)
    out = {"mean_diff": float(diff.mean()), "wins_a": int((diff > 0).sum()), "wins_b": int((diff < 0).sum()), "ties": int((diff == 0).sum())}
    var = diff.var(ddof=1) if k > 1 else 0.0
    if var > 0:
        n_test = np.mean([f["n_test"] for f in a.folds])
        n_train = np.mean([f["n_train"] for f in a.folds])
        t = diff.mean() / np.sqrt((1 / k + n_test / n_train) * var)
        out["t"] = float(t)
        out["p_value"] = float(2 * stats.t.sf(abs(t), df=k - 1))
    else:
        out["t"], out["p_value"] = 0.0, 1.0
    return out


def subset_metrics(oof: list[dict[str, int]], samples: set[int], n_classes: int) -> dict[str, Any] | None:
    """Metrics of out-of-fold predictions restricted to some samples (pooled over repeats)."""
    rows = [r for r in oof if r["sample"] in samples]
    if not rows:
        return None
    y_true = np.array([r["y_true"] for r in rows])
    y_pred = np.array([r["y_pred"] for r in rows])
    return score(y_true, y_pred, n_classes)


def feature_importance(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
    splitter: RepeatedStratifiedKFold,
    params: dict[str, Any],
    seed: int,
    permutation_repeats: int,
) -> pd.DataFrame:
    """Model-agnostic permutation importance (+ the model's native importance).

    Permutation importance (drop in macro-F1 when a column is shuffled) is measured on
    held-out folds of the first CV repeat, so it is directly comparable across models.
    The native column is impurity importance for the forest and mean |coefficient|
    (standardised features) for logistic regression, from a fit on all data.
    """
    template = build_pipeline(name, list(X.columns), params, seed)
    n_splits = splitter.cvargs["n_splits"]
    drops = []
    for i, (train, test) in enumerate(splitter.split(X, y)):
        if i >= n_splits:  # first repeat only
            break
        pipe = clone(template).fit(X.iloc[train], y[train])
        res = permutation_importance(
            pipe, X.iloc[test], y[test], scoring="f1_macro",
            n_repeats=permutation_repeats, random_state=seed, n_jobs=1,
        )
        drops.append(res.importances_mean)
    drops = np.array(drops)

    full = clone(template).fit(X, y)
    model = full[-1]
    if hasattr(model, "feature_importances_"):
        native = pd.Series(model.feature_importances_, index=X.columns)
    else:
        names = list(full[:-1].get_feature_names_out())
        native = pd.Series(np.abs(model.coef_).mean(axis=0), index=names).reindex(X.columns)

    return pd.DataFrame({
        "feature": X.columns,
        "permutation_mean": drops.mean(axis=0),
        "permutation_std": drops.std(axis=0, ddof=1) if len(drops) > 1 else 0.0,
        "native": native.values,
    }).sort_values("permutation_mean", ascending=False).reset_index(drop=True)


def logistic_coefficients(
    X: pd.DataFrame, y: np.ndarray, class_names: list[str], params: dict[str, Any], seed: int
) -> pd.DataFrame:
    """Per-class coefficients (standardised features) of a logistic regression fit on all data."""
    pipe = build_pipeline("logistic_regression", list(X.columns), params, seed).fit(X, y)
    names = list(pipe[:-1].get_feature_names_out())
    coef = pipe[-1].coef_
    if coef.shape[0] == 1:  # binary: sklearn stores a single row
        coef = np.vstack([-coef[0], coef[0]])
    return pd.DataFrame(coef.T, index=names, columns=class_names).reindex(X.columns)

