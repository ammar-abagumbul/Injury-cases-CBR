"""Experiment 6 — injury severity prediction with classic ML.

Features are built from LLM-extracted ``Case`` JSON files (ICD-11 injuries,
VMIS-tagged treatments, hospital / sick-leave figures, demographics, losses).
The label is the human-annotated overall severity from the labelled CSV
(``injury_type``), because humans judge the overall category better than the LLM.

Two models are trained on identical stratified repeated-CV folds and compared:

* Random Forest        — non-linear, handles mixed / missing features, interactions
* Logistic Regression  — linear and additive, readable coefficients

A majority-class baseline and the LLM's own ``overall_category`` (where it exists)
are reported for reference. Outputs: metrics, feature importance, confusion
matrices and a few distribution plots. See ``config.yaml`` for the options.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, override

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.table import Table

from hklii_psla.experiments.experiment import BaseExperiment
from hklii_psla.experiments.experiment6 import plots
from hklii_psla.experiments.experiment6.dataset import (
    JoinReport,
    LabelledCase,
    load_labelled_cases,
    tier_lookup,
)
from hklii_psla.experiments.experiment6.features import (
    FEATURE_GROUPS,
    build_feature_table,
    llm_overall_tier,
)
from hklii_psla.experiments.experiment6.modeling import (
    METRICS,
    MODEL_LABELS,
    CVResult,
    compare_models,
    cross_validate,
    feature_importance,
    logistic_coefficients,
    make_splitter,
    score,
    subset_metrics,
)

console = Console()

COMPARED_MODELS = ("random_forest", "logistic_regression")


class Exp6Config(BaseModel):
    id: str
    cases_path: str
    cases_glob: str = "*.json"
    labels_path: str
    label_column: str = "injury_type"
    # ordered low -> high; each tier lists the raw (lowercase) labels merged into it
    tiers: dict[str, list[str]] = Field(default_factory=lambda: {
        "non-serious": ["nonserious", "non-serious"],
        "serious": ["serious"],
        "substantial+": ["substantial", "gross", "disaster"],
    })
    output_dir: str
    feature_sets: dict[str, list[str]] = Field(default_factory=lambda: {
        "core": ["injuries", "treatment", "hospital_sick", "demographics"],
        "all": list(FEATURE_GROUPS),
    })
    primary_feature_set: str = "all"
    models: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {name: {} for name in COMPARED_MODELS}
    )
    n_splits: int = 5
    n_repeats: int = 5
    seed: int = 42
    min_feature_support: int = 3
    permutation_repeats: int = 5
    top_n_features: int = 20
    distribution_features: int = 6

    @model_validator(mode="after")
    def _check(self) -> Exp6Config:
        unknown_models = set(self.models) - set(COMPARED_MODELS)
        if unknown_models:
            raise ValueError(f"Unknown models {sorted(unknown_models)}; choose from {list(COMPARED_MODELS)}")
        for name, groups in self.feature_sets.items():
            bad = set(groups) - set(FEATURE_GROUPS)
            if bad:
                raise ValueError(f"feature_sets[{name!r}] has unknown groups {sorted(bad)}; use {list(FEATURE_GROUPS)}")
        if self.primary_feature_set not in self.feature_sets:
            raise ValueError(f"primary_feature_set {self.primary_feature_set!r} not in feature_sets")
        if len(self.tiers) < 2:
            raise ValueError("Need at least two tiers")
        return self


class SeverityPredictionExperiment(BaseExperiment[Exp6Config], experiment_id="severity-prediction"):

    _config: Exp6Config
    _config_model: ClassVar[type[BaseModel]] = Exp6Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._result: dict[str, Any] | None = None
        self._frames: dict[str, Any] = {}

    # -- pipeline ---------------------------------------------------------
    @override
    def run(self) -> None:
        cfg = self._config
        tier_names = list(cfg.tiers)

        labelled, report = load_labelled_cases(
            Path(cfg.cases_path), cfg.cases_glob, Path(cfg.labels_path), cfg.label_column, cfg.tiers
        )
        self._print_join(report, len(labelled))
        if not labelled:
            raise RuntimeError(
                "No extracted case matched a labelled CSV row. Check cases_path / labels_path "
                f"({cfg.cases_path!r}, {cfg.labels_path!r}) and the unmatched list above."
            )

        # tiers with no labelled case cannot be learned or scored: drop them (loudly)
        present = {lc.tier for lc in labelled}
        empty = [t for t in tier_names if t not in present]
        if empty:
            console.print(f"[yellow]No labelled case in tier(s) {empty}; continuing without them.[/yellow]")
            tier_names = [t for t in tier_names if t in present]
        if len(tier_names) < 2:
            raise RuntimeError(f"Only one tier has labelled cases ({tier_names}); nothing to predict.")

        X, groups = build_feature_table([lc.case for lc in labelled], cfg.min_feature_support)
        y = np.array([tier_names.index(lc.tier) for lc in labelled])
        tier_counts = {t: int((y == i).sum()) for i, t in enumerate(tier_names)}
        raw_counts = self._count_raw(labelled)
        console.print(f"[cyan]Labelled cases: {len(labelled)}  tiers: {tier_counts}  features: {X.shape[1]}[/cyan]")

        splitter, n_splits = make_splitter(y, cfg.n_splits, cfg.n_repeats, cfg.seed)
        if n_splits != cfg.n_splits:
            console.print(f"[yellow]Rarest tier is small: using {n_splits}-fold CV instead of {cfg.n_splits}.[/yellow]")

        # ---- cross-validation: every feature set x model on the same folds
        cv: dict[str, dict[str, CVResult]] = {}
        columns_of: dict[str, list[str]] = {}
        for set_name, group_names in cfg.feature_sets.items():
            columns = [c for g in group_names for c in groups[g]]
            columns_of[set_name] = columns
            cv[set_name] = {}
            for model in cfg.models:
                console.print(f"[dim]CV: {set_name} / {MODEL_LABELS[model]} ({len(columns)} features)[/dim]")
                cv[set_name][model] = cross_validate(
                    model, X[columns], y, len(tier_names), splitter, cfg.models[model], cfg.seed
                )
        majority = cross_validate("majority", X, y, len(tier_names), splitter, {}, cfg.seed)

        # ---- paired comparison of the two models
        comparison: dict[str, dict[str, Any]] = {}
        if all(m in cfg.models for m in COMPARED_MODELS):
            for set_name in cfg.feature_sets:
                rf, lr = (cv[set_name][m] for m in COMPARED_MODELS)
                comparison[set_name] = {
                    metric: compare_models(rf, lr, metric, n_splits) for metric in ("macro_f1", "qwk")
                }

        # ---- LLM overall_category baseline (only where it exists)
        llm_baseline = self._llm_baseline(labelled, y, tier_names, cv)

        # ---- importance and coefficients on the primary feature set
        primary = cfg.primary_feature_set
        Xp = X[columns_of[primary]]
        importances: dict[str, pd.DataFrame] = {}
        for model in cfg.models:
            console.print(f"[dim]Permutation importance: {MODEL_LABELS[model]}[/dim]")
            importances[model] = feature_importance(
                model, Xp, y, len(tier_names), splitter, cfg.models[model], cfg.seed, cfg.permutation_repeats
            )
        coefs = None
        if "logistic_regression" in cfg.models:
            coefs = logistic_coefficients(Xp, y, tier_names, cfg.models["logistic_regression"], cfg.seed)

        self._frames = {
            "X": X, "y": y, "labelled": labelled, "groups": groups, "cv": cv, "majority": majority,
            "importances": importances, "coefs": coefs, "tier_counts": tier_counts, "raw_counts": raw_counts,
            "columns_of": columns_of,
        }
        self._result = {
            "id": cfg.id,
            "config": cfg.model_dump(),
            "dataset": {
                "n_labelled_cases": len(labelled),
                "tiers": tier_names,
                "tier_counts": tier_counts,
                "raw_label_counts": raw_counts,
                "n_features_total": int(X.shape[1]),
                "features_per_group": {g: len(c) for g, c in groups.items()},
                "n_splits_used": n_splits,
                "n_repeats": cfg.n_repeats,
                "join": report.to_dict(),
            },
            "results": {
                set_name: {m: r.summary() for m, r in models.items()} for set_name, models in cv.items()
            },
            "baselines": {"majority": majority.summary(), "llm_overall_category": llm_baseline},
            "model_comparison": comparison,
            "top_features": {
                m: df.head(cfg.top_n_features)[["feature", "permutation_mean", "permutation_std", "native"]]
                .to_dict("records") for m, df in importances.items()
            },
        }
        self._display(cv, majority, comparison, llm_baseline, importances, tier_names)

    # -- helpers ----------------------------------------------------------
    def _count_raw(self, labelled: list[LabelledCase]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lc in labelled:
            counts[lc.raw_label] = counts.get(lc.raw_label, 0) + 1
        order = [r for raws in self._config.tiers.values() for r in raws if r in counts]
        return {r: counts[r] for r in order}

    def _llm_baseline(
        self,
        labelled: list[LabelledCase],
        y: np.ndarray,
        tier_names: list[str],
        cv: dict[str, dict[str, CVResult]],
    ) -> dict[str, Any]:
        lookup = tier_lookup(self._config.tiers)
        idx, preds = [], []
        for i, lc in enumerate(labelled):
            raw = llm_overall_tier(lc.case)
            # the LLM may name a tier that has no labelled case (and was dropped): skip it
            if raw is not None and lookup.get(raw) in tier_names:
                idx.append(i)
                preds.append(tier_names.index(lookup[raw]))
        out: dict[str, Any] = {"n_cases_with_value": len(idx), "n_cases": len(labelled)}
        if not idx:
            return out
        out["metrics"] = score(y[idx], np.array(preds), len(tier_names))
        subset = set(idx)
        out["models_on_same_cases"] = {
            set_name: {m: subset_metrics(r.oof, subset, len(tier_names)) for m, r in models.items()}
            for set_name, models in cv.items()
        }
        return out

    def _print_join(self, report: JoinReport, n_matched: int) -> None:
        table = Table(title="Case ↔ label join")
        table.add_column("", style="cyan")
        table.add_column("count", justify="right")
        table.add_row("extracted case files", str(report.n_case_files))
        table.add_row("matched to a human label", str(n_matched))
        table.add_row("no matching CSV row", str(len(report.unmatched_cases)))
        table.add_row("conflicting labels (dropped)", str(len(report.conflicting_cases)))
        table.add_row("duplicates of a matched row (dropped)", str(len(report.duplicate_cases)))
        table.add_row("unreadable files", str(report.n_unreadable))
        table.add_row("CSV rows without a usable label", f"{report.n_csv_rows_without_usable_label}/{report.n_csv_rows}")
        console.print(table)
        if report.unreadable_files:
            console.print("[red]Unreadable case files (first 3):[/red]")
            for line in report.unreadable_files[:3]:
                console.print(f"  [red]{line}[/red]")
        if report.unmatched_cases:
            console.print(f"[yellow]Unmatched: {', '.join(report.unmatched_cases[:15])}"
                          + (" ..." if len(report.unmatched_cases) > 15 else "") + "[/yellow]")

    def _display(
        self,
        cv: dict[str, dict[str, CVResult]],
        majority: CVResult,
        comparison: dict[str, dict[str, Any]],
        llm_baseline: dict[str, Any],
        importances: dict[str, pd.DataFrame],
        tier_names: list[str],
    ) -> None:
        def pm(d: dict[str, float]) -> str:
            return f"{d['mean']:.3f} ± {d['std']:.3f}"

        table = Table(title="Experiment 6 — severity prediction (repeated stratified CV)")
        table.add_column("Feature set", style="cyan")
        table.add_column("Model", style="green")
        for metric in METRICS:
            table.add_column(metric.replace("_", " "), justify="right")
        for t in tier_names:
            table.add_column(f"recall {t}", justify="right")

        def add(set_name: str, label: str, summary: dict[str, Any]) -> None:
            table.add_row(
                set_name, label, *(pm(summary[m]) for m in METRICS),
                *(f"{r:.2f}" for r in summary["recall"]["mean"]),
            )

        add("—", MODEL_LABELS["majority"], majority.summary())
        for set_name, models in cv.items():
            for model, result in models.items():
                add(set_name, MODEL_LABELS[model], result.summary())
        console.print(table)

        if comparison:
            cmp = Table(title="Random Forest − Logistic Regression (paired folds, Nadeau-Bengio corrected t-test)")
            for col in ("Feature set", "Metric", "mean diff", "RF wins / LR wins / ties", "p-value"):
                cmp.add_column(col)
            for set_name, metrics in comparison.items():
                for metric, c in metrics.items():
                    cmp.add_row(set_name, metric, f"{c['mean_diff']:+.3f}",
                                f"{c['wins_a']} / {c['wins_b']} / {c['ties']}", f"{c['p_value']:.3f}")
            console.print(cmp)

        if "metrics" in llm_baseline:
            m = llm_baseline["metrics"]
            console.print(
                f"LLM overall_category baseline ({llm_baseline['n_cases_with_value']}/{llm_baseline['n_cases']} "
                f"cases have one): macro-F1 {m['macro_f1']:.3f}, QWK {m['qwk']:.3f}, accuracy {m['accuracy']:.3f}"
            )
        else:
            console.print("[dim]No case has an LLM overall_category to use as a baseline.[/dim]")

        for model, df in importances.items():
            imp = Table(title=f"Top features — {MODEL_LABELS[model]} (primary set: {self._config.primary_feature_set})")
            for col in ("Feature", "Δ macro-F1 when shuffled", "native importance"):
                imp.add_column(col, justify="left" if col == "Feature" else "right")
            for _, row in df.head(10).iterrows():
                imp.add_row(row["feature"], f"{row['permutation_mean']:+.4f} ± {row['permutation_std']:.4f}", f"{row['native']:.4f}")
            console.print(imp)

    # -- persistence ------------------------------------------------------
    @override
    def save_results(self) -> None:
        if self._result is None:
            raise RuntimeError("run() must be called before save_results()")

        cfg = self._config
        out = Path(cfg.output_dir)
        plot_dir = out / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        f = self._frames
        X: pd.DataFrame = f["X"]
        tier_names = list(f["tier_counts"])
        labelled: list[LabelledCase] = f["labelled"]
        cv: dict[str, dict[str, CVResult]] = f["cv"]

        # dataset: the exact table the models saw
        dataset = X.copy()
        dataset.insert(0, "case_id", [lc.case_id for lc in labelled])
        dataset.insert(1, "raw_label", [lc.raw_label for lc in labelled])
        dataset.insert(2, "tier", [lc.tier for lc in labelled])
        dataset.to_csv(out / "dataset.csv", index=False)

        # per-fold metrics and out-of-fold predictions
        fold_rows, oof_rows = [], []
        runs = [("—", "majority", f["majority"])] + [
            (s, m, r) for s, models in cv.items() for m, r in models.items()
        ]
        for set_name, model, result in runs:
            for fold in result.folds:
                fold_rows.append({"feature_set": set_name, "model": model, **{
                    k: v for k, v in fold.items() if k != "recall"
                }, **{f"recall_{t}": r for t, r in zip(tier_names, fold["recall"])}})
            for row in result.oof:
                oof_rows.append({
                    "feature_set": set_name, "model": model,
                    "case_id": labelled[row["sample"]].case_id, "repeat": row["repeat"],
                    "true": tier_names[row["y_true"]], "pred": tier_names[row["y_pred"]],
                })
        pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
        pd.DataFrame(oof_rows).to_csv(out / "oof_predictions.csv", index=False)

        for model, df in f["importances"].items():
            df.to_csv(out / f"feature_importance_{model}.csv", index=False)
        if f["coefs"] is not None:
            f["coefs"].to_csv(out / "logistic_coefficients.csv")

        # figures
        plots.label_distribution(f["raw_counts"], f["tier_counts"], plot_dir / "label_distribution.png")
        plots.metrics_comparison(
            {s: {"majority": f["majority"].summary(), **{m: r.summary() for m, r in models.items()}}
             for s, models in cv.items()},
            ("macro_f1", "qwk", "balanced_accuracy"), plot_dir / "metrics_comparison.png",
        )
        plots.confusion_matrices(
            {m: r.normalised_confusion() for m, r in cv[cfg.primary_feature_set].items()},
            tier_names, plot_dir / "confusion_matrices.png",
            f"Feature set: {cfg.primary_feature_set}",
        )
        plots.importance_bars(
            f["importances"], cfg.top_n_features, plot_dir / "feature_importance.png",
            f"Permutation importance (primary set: {cfg.primary_feature_set})",
        )
        if f["coefs"] is not None:
            order = f["importances"]["logistic_regression"]["feature"].tolist()
            plots.coefficient_heatmap(f["coefs"], order, min(cfg.top_n_features, 15), plot_dir / "logistic_coefficients.png")

        ranking = next(iter(f["importances"].values()))
        for model, df in f["importances"].items():  # prefer the forest's ranking when present
            if model == "random_forest":
                ranking = df
        top = [c for c in ranking["feature"].head(cfg.distribution_features)]
        plots.feature_distributions(
            X, pd.Series([lc.tier for lc in labelled]), top, tier_names, plot_dir / "feature_distributions.png"
        )

        with (out / "results.json").open("w", encoding="utf-8") as fh:
            json.dump(self._result, fh, indent=2, ensure_ascii=False, default=float)

        console.print(f"\n[bold green]Results saved to {out}[/bold green]")
        console.print(f"[green]Figures: {plot_dir}[/green]")
