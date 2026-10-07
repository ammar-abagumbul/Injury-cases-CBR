"""Experiment 7 — feature-query CBR retrieval with PSLA-aware evaluation.

A judge supplies case features; the system retrieves the most similar corpus
cases with case-based reasoning, and the experiment measures how close the
retrieved PSLA compensations are to the query's.

Two modes:

* **Leave-one-out benchmark** (no ``query_file``): every corpus case with a
  usable PSLA is used as a feature query against the rest.  We report PSLA
  closeness, PSLA-band IR metrics (precision/recall/NDCG@k with random
  baselines) and feature overlap, then tune the CBR similarity weights on a
  train split and evaluate the tuned weights on a held-out test split.
* **Ad-hoc judge query** (``query_file``): load a YAML/JSON feature query,
  optionally restrict candidates to a real-HKD PSLA band, retrieve and save.

All awards are deflated to a common base year with ``data/price_index.csv``
before any comparison.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, override

import numpy as np
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from hklii_psla.experiments.experiment import BaseExperiment
from hklii_psla.experiments.experiment4.icd_similarity import IcdTaxonomy
from hklii_psla.experiments.experiment7.cbr_model import (
    DEFAULT_ATTRIBUTES,
    CorpusCBRModel,
    build_corpus_model,
    clinical_gate_mask,
    composite_similarities,
)
from hklii_psla.experiments.experiment7.evaluation import (
    ClinicalRelevance,
    QueryEvaluation,
    aggregate_metrics,
    clinical_relevance,
    evaluate_query,
    summarise_evaluations,
)
from hklii_psla.experiments.experiment7.features import (
    DEFAULT_CORPUS_DIR,
    CorpusCase,
    graded_relevance,
    is_relevant,
    load_corpus,
)
from hklii_psla.experiments.experiment7.features import (
    summarise as summarise_corpus,
)
from hklii_psla.experiments.experiment7.price_index import (
    DEFAULT_PRICE_INDEX_PATH,
    PriceIndex,
)
from hklii_psla.experiments.experiment7.query import FeatureQuery, load_query
from hklii_psla.experiments.experiment7.retrieval import CBRFeatureRetriever

console = Console()

_WEIGHT_LOW, _WEIGHT_HIGH = 0.1, 3.0


class Exp7Config(BaseModel):
    id: str
    output_dir: str
    case_base_dir: str = str(DEFAULT_CORPUS_DIR)
    k: int = 10
    psla_tolerance: float = 0.5
    weights: dict[str, float] | None = None
    attributes: list[str] | None = None
    missing_policy: Literal["available", "special"] = "available"
    # Clinical-first scoring: share of the composite carried by the clinical
    # (injury + loss) group; the remainder is auxiliary.
    clinical_alpha: float = 0.7
    # Clinical relevance label (loose, independent of PSLA): graded gain is
    # injury_weight * injury_sim + (1 - injury_weight) * loss_sim, and a
    # candidate is "relevant" at or above clinical_threshold.
    clinical_threshold: float = 0.15
    clinical_injury_weight: float = 0.75
    # Stage-1 loose gate: a candidate must clear this graded injury similarity
    # (only enforced when the query carries injuries).
    min_injury_similarity: float = 0.1
    tune: bool = True
    tune_trials: int = 40
    tune_objective: Literal["clinical_ndcg", "ndcg"] = "clinical_ndcg"
    train_fraction: float = 0.6
    seed: int = 42
    max_queries: int | None = None
    query_file: str | None = None
    weights_file: str | None = None
    price_index_path: str | None = None
    base_year: int | None = None
    band_low: float | None = None
    band_high: float | None = None
    require_psla_candidates: bool = True


def _weight_array(
    names: Sequence[str], weights: dict[str, float] | None
) -> np.ndarray:
    return np.asarray(
        [float((weights or {}).get(name, 1.0)) for name in names],
        dtype=np.float64,
    )


def _sample_weights(
    rng: np.random.Generator, names: Sequence[str]
) -> dict[str, float]:
    values = np.exp(
        rng.uniform(np.log(_WEIGHT_LOW), np.log(_WEIGHT_HIGH), size=len(names))
    )
    return {name: float(v) for name, v in zip(names, values)}


class FeatureQueryRetrievalExperiment(
    BaseExperiment[Exp7Config], experiment_id="feature-query-retrieval"
):

    _config: Exp7Config
    _config_model: ClassVar[type[BaseModel]] = Exp7Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._result: dict[str, Any] | None = None
        self._per_query: list[QueryEvaluation] = []
        self._taxonomy = IcdTaxonomy.from_json()

    # -- pipeline ---------------------------------------------------------
    @override
    def run(self) -> None:
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        price_index = PriceIndex.from_csv(
            self._config.price_index_path or DEFAULT_PRICE_INDEX_PATH,
            base_year=self._config.base_year,
        )
        cases = load_corpus(self._config.case_base_dir, price_index)

        console.print(
            f"[bold]Loaded {len(cases)} cases from "
            f"{self._config.case_base_dir}[/bold]"
        )
        corpus_summary = summarise_corpus(cases)
        console.print(
            f"  with PSLA: [cyan]{corpus_summary['n_with_psla']}[/cyan]  "
            f"with year: [cyan]{corpus_summary['n_with_year']}[/cyan]  "
            f"median real PSLA: "
            f"[cyan]{_fmt_money(corpus_summary['median_real_psla'])}[/cyan]"
        )

        if self._config.query_file:
            self._result = self._run_ad_hoc_query(cases)
        else:
            self._result = self._run_benchmark(cases, corpus_summary)

    # -- model helper -----------------------------------------------------
    def _build_model(self, cases: Sequence[CorpusCase]) -> CorpusCBRModel:
        return build_corpus_model(
            cases,
            attributes=self._config.attributes or DEFAULT_ATTRIBUTES,
            missing_policy=self._config.missing_policy,
        )

    def _resolved_weights(self) -> dict[str, float] | None:
        """Config weights, else a persisted ``tuned_weights.json``, else None."""
        if self._config.weights:
            return {k: float(v) for k, v in self._config.weights.items()}
        path: Path | None = None
        if self._config.weights_file:
            path = Path(self._config.weights_file)
        else:
            candidate = Path(self._config.output_dir) / "tuned_weights.json"
            path = candidate if candidate.exists() else None
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        weights = data.get("weights", data)
        return {str(k): float(v) for k, v in weights.items()}

    # -- ad-hoc judge query ----------------------------------------------
    def _run_ad_hoc_query(self, cases: list[CorpusCase]) -> dict[str, Any]:
        query = load_query(self._config.query_file)  # type: ignore[arg-type]
        band = self._resolve_band(query)
        model = self._build_model(cases)
        retriever = CBRFeatureRetriever(model)
        weights = query.weights or self._resolved_weights()
        result = retriever.retrieve(
            query.to_corpus_case(self._taxonomy),
            k=query.k or self._config.k,
            weights=weights,
            psla_band=band,
            require_psla=self._config.require_psla_candidates,
            clinical_alpha=self._config.clinical_alpha,
            min_injury_similarity=self._config.min_injury_similarity,
        )

        self._display_query_result(query, result)
        query_case = query.to_corpus_case(self._taxonomy)
        return {
            "mode": "ad_hoc_query",
            "query": query.model_dump(mode="json"),
            "resolved_injuries": list(query_case.injuries),
            "query_severity": {
                "extracted": query_case.overall_category,
                "from_award": query_case.award_severity,
            },
            "weights": result.weights,
            "clinical_alpha": self._config.clinical_alpha,
            "min_injury_similarity": self._config.min_injury_similarity,
            "psla_band": result.psla_band,
            "results": [
                {
                    "case_id": r.case_id,
                    "similarity": r.similarity,
                    "psla_nominal": r.psla_nominal,
                    "psla_real": r.psla_real,
                    "year": r.year,
                    "extracted_severity": r.extracted_severity,
                    "award_severity": r.award_severity,
                }
                for r in result.results
            ],
        }

    def _resolve_band(
        self, query: FeatureQuery
    ) -> tuple[float, float] | None:
        if query.psla_band is not None:
            return (float(query.psla_band[0]), float(query.psla_band[1]))
        low = self._config.band_low
        high = self._config.band_high
        if low is not None and high is not None:
            return (float(low), float(high))
        if query.psla_target_real is not None:
            tol = 1.0 + self._config.psla_tolerance
            target = float(query.psla_target_real)
            return (target / tol, target * tol)
        return None

    # -- leave-one-out benchmark -----------------------------------------
    def _run_benchmark(
        self, cases: list[CorpusCase], corpus_summary: dict[str, Any]
    ) -> dict[str, Any]:
        pool = [c for c in cases if c.has_psla]
        if len(pool) < 2:
            raise RuntimeError("Not enough cases with a usable PSLA for the benchmark")

        rng = np.random.default_rng(self._config.seed)
        if self._config.max_queries is not None:
            pool = list(
                rng.choice(
                    pool,
                    size=min(self._config.max_queries, len(pool)),
                    replace=False,
                )
            )

        pool_ids = [c.case_id for c in pool]
        index_of = {cid: i for i, cid in enumerate(pool_ids)}
        queries = pool

        console.print(
            f"[bold]Building CBR model over {len(cases)} cases "
            f"({len(pool)} in the PSLA benchmark pool)...[/bold]"
        )
        model = self._build_model(cases)
        attribute_names = model.attribute_names
        console.print(
            f"  similarity attributes: [cyan]{', '.join(attribute_names)}[/cyan]"
        )

        console.print("[bold]Caching per-attribute similarities...[/bold]")
        S, known = model.per_attribute_similarities(queries, candidate_ids=pool_ids)

        # Leave-one-out mask: exclude each query's own pool entry.
        mask = np.ones((len(queries), len(pool)), dtype=bool)
        for qi, q in enumerate(queries):
            mask[qi, index_of[q.case_id]] = False

        gains, relevant = self._gain_arrays(queries, pool, mask)
        clinical = clinical_relevance(
            S,
            known,
            attribute_names,
            threshold=self._config.clinical_threshold,
            injury_weight=self._config.clinical_injury_weight,
        )
        query_has_injuries = np.asarray([len(q.injuries) > 0 for q in queries])

        default_weights = {
            name: float((self._resolved_weights() or {}).get(name, 1.0))
            for name in attribute_names
        }

        console.print("[bold]Evaluating default weights (leave-one-out)...[/bold]")
        default_evals = self._evaluate_all(
            queries, pool, S, known, mask, clinical, query_has_injuries,
            default_weights, attribute_names,
        )
        self._per_query = default_evals
        default_summary = summarise_evaluations(default_evals)

        tuning: dict[str, Any] | None = None
        if self._config.tune:
            tuning = self._tune(
                queries, pool, S, known, gains, relevant, clinical,
                query_has_injuries, mask, attribute_names, default_weights, rng,
            )

        result: dict[str, Any] = {
            "mode": "leave_one_out_benchmark",
            "id": self._config.id,
            "config": self._config.model_dump(mode="json"),
            "corpus": corpus_summary,
            "attribute_names": list(attribute_names),
            "default_weights": default_weights,
            "benchmark": {
                "n_queries": len(queries),
                "n_pool": len(pool),
                "default": default_summary,
                "tuning": tuning,
            },
        }
        self._display_benchmark(result)
        return result

    # -- arrays / evaluation ---------------------------------------------
    def _gain_arrays(
        self,
        queries: Sequence[CorpusCase],
        pool: Sequence[CorpusCase],
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        gains = np.asarray(
            [
                [
                    graded_relevance(q.psla_real, c.psla_real, self._config.psla_tolerance)
                    for c in pool
                ]
                for q in queries
            ],
            dtype=np.float64,
        )
        relevant = np.asarray(
            [
                [
                    is_relevant(q.psla_real, c.psla_real, self._config.psla_tolerance)
                    for c in pool
                ]
                for q in queries
            ],
            dtype=bool,
        )
        return np.where(mask, gains, 0.0), np.where(mask, relevant, False)

    def _global_similarities(
        self,
        S: np.ndarray,
        known: np.ndarray,
        mask: np.ndarray,
        weights: dict[str, float],
        attribute_names: Sequence[str],
        query_has_injuries: np.ndarray,
    ) -> np.ndarray:
        vec = _weight_array(attribute_names, weights)
        G = composite_similarities(
            S, known, vec, attribute_names,
            clinical_alpha=self._config.clinical_alpha,
        )
        gate = clinical_gate_mask(
            S, known, attribute_names,
            min_injury_similarity=self._config.min_injury_similarity,
            query_has_injuries=query_has_injuries,
        )
        return np.where(mask & gate, G, -np.inf)

    def _evaluate_all(
        self,
        queries: Sequence[CorpusCase],
        pool: Sequence[CorpusCase],
        S: np.ndarray,
        known: np.ndarray,
        mask: np.ndarray,
        clinical: ClinicalRelevance,
        query_has_injuries: np.ndarray,
        weights: dict[str, float],
        attribute_names: Sequence[str],
    ) -> list[QueryEvaluation]:
        G = self._global_similarities(
            S, known, mask, weights, attribute_names, query_has_injuries
        )
        evaluations: list[QueryEvaluation] = []
        for qi, query in enumerate(queries):
            evaluations.append(
                evaluate_query(
                    query,
                    pool,
                    G[qi],
                    k=self._config.k,
                    tolerance=self._config.psla_tolerance,
                    taxonomy=self._taxonomy,
                    mask=mask[qi],
                    clinical=clinical.select(qi),
                )
            )
        return evaluations

    def _tune(
        self,
        queries: Sequence[CorpusCase],
        pool: Sequence[CorpusCase],
        S: np.ndarray,
        known: np.ndarray,
        gains: np.ndarray,
        relevant: np.ndarray,
        clinical: ClinicalRelevance,
        query_has_injuries: np.ndarray,
        mask: np.ndarray,
        attribute_names: Sequence[str],
        default_weights: dict[str, float],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        q = len(queries)
        order = rng.permutation(q)
        n_train = max(2, int(q * self._config.train_fraction))
        train_idx = np.sort(order[:n_train])
        test_idx = np.sort(order[n_train:])

        S_train, known_train = S[train_idx], known[train_idx]
        gains_train, rel_train, mask_train = (
            gains[train_idx],
            relevant[train_idx],
            mask[train_idx],
        )
        clinical_train = clinical.select(train_idx)
        qhi_train = query_has_injuries[train_idx]

        objective = (
            "mean_clinical_ndcg_at_k"
            if self._config.tune_objective == "clinical_ndcg"
            else "mean_ndcg_at_k"
        )

        def score(weights: dict[str, float]) -> float:
            G = self._global_similarities(
                S_train, known_train, mask_train, weights, attribute_names,
                qhi_train,
            )
            return aggregate_metrics(
                G, gains_train, rel_train, k=self._config.k,
                clinical=clinical_train,
            )[objective]

        history: list[dict[str, Any]] = []
        default_score = score(default_weights)
        history.append(
            {"weights": dict(default_weights), "train_objective": default_score}
        )
        best_weights = dict(default_weights)
        best_score = default_score

        console.print(
            f"[bold]Tuning weights ({self._config.tune_trials} trials, "
            f"objective={self._config.tune_objective}) on "
            f"{len(train_idx)} queries...[/bold]"
        )
        for _ in range(self._config.tune_trials):
            candidate = _sample_weights(rng, attribute_names)
            value = score(candidate)
            history.append({"weights": candidate, "train_objective": value})
            if value > best_score:
                best_score = value
                best_weights = candidate

        test_summary_default = self._summary_for_subset(
            queries, pool, S, known, mask, clinical, query_has_injuries,
            test_idx, default_weights, attribute_names,
        )
        test_summary_best = self._summary_for_subset(
            queries, pool, S, known, mask, clinical, query_has_injuries,
            test_idx, best_weights, attribute_names,
        )

        console.print(
            f"  default train {self._config.tune_objective}: "
            f"[yellow]{default_score:.3f}[/yellow]  "
            f"best train: [green]{best_score:.3f}[/green]"
        )
        console.print(
            f"  held-out clinical NDCG — default: "
            f"[yellow]{test_summary_default['mean_clinical_ndcg_at_k']:.3f}[/yellow]  "
            f"tuned: [green]{test_summary_best['mean_clinical_ndcg_at_k']:.3f}[/green]"
        )
        console.print(
            f"  held-out PSLA-band NDCG — default: "
            f"[yellow]{test_summary_default['mean_ndcg_at_k']:.3f}[/yellow]  "
            f"tuned: [green]{test_summary_best['mean_ndcg_at_k']:.3f}[/green]"
        )

        return {
            "objective": self._config.tune_objective,
            "objective_metric": objective,
            "trials": self._config.tune_trials,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "best_weights": best_weights,
            "default_train_objective": default_score,
            "best_train_objective": best_score,
            "test_default": test_summary_default,
            "test_best": test_summary_best,
            "history": history,
        }

    def _summary_for_subset(
        self,
        queries: Sequence[CorpusCase],
        pool: Sequence[CorpusCase],
        S: np.ndarray,
        known: np.ndarray,
        mask: np.ndarray,
        clinical: ClinicalRelevance,
        query_has_injuries: np.ndarray,
        subset_idx: np.ndarray,
        weights: dict[str, float],
        attribute_names: Sequence[str],
    ) -> dict[str, Any]:
        S_sub, known_sub, mask_sub = S[subset_idx], known[subset_idx], mask[subset_idx]
        G = self._global_similarities(
            S_sub, known_sub, mask_sub, weights, attribute_names,
            query_has_injuries[subset_idx],
        )
        gains, relevant = self._gain_arrays(
            [queries[i] for i in subset_idx], pool, mask_sub
        )
        return aggregate_metrics(
            G, gains, relevant, k=self._config.k,
            clinical=clinical.select(subset_idx),
        )

    # -- display ----------------------------------------------------------
    def _display_query_result(self, query: FeatureQuery, result: Any) -> None:
        table = Table(
            title=f"Experiment 7 — query '{query.query_id}' "
            f"({len(result.results)} results)"
        )
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Case", style="green")
        table.add_column("Similarity", justify="right", style="magenta")
        table.add_column("Year", justify="right")
        table.add_column("PSLA (nominal)", justify="right", style="yellow")
        table.add_column("PSLA (real)", justify="right", style="yellow")
        for i, r in enumerate(result.results):
            table.add_row(
                str(i + 1),
                r.case_id,
                f"{r.similarity:.3f}",
                "" if r.year is None else str(r.year),
                _fmt_money(r.psla_nominal),
                _fmt_money(r.psla_real),
            )
        console.print(table)
        if result.psla_band:
            console.print(
                f"[cyan]PSLA band (real HKD): "
                f"{_fmt_money(result.psla_band[0])} – "
                f"{_fmt_money(result.psla_band[1])}[/cyan]"
            )

    def _display_benchmark(self, result: dict[str, Any]) -> None:
        summary = result["benchmark"]["default"]
        table = Table(title="Experiment 7 — Leave-one-out benchmark (default weights)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        rows = [
            ("Queries", str(result["benchmark"]["n_queries"])),
            ("Candidate pool", str(result["benchmark"]["n_pool"])),
            ("Mean clinical NDCG@k", f"{summary['mean_clinical_ndcg_at_k']:.3f}"),
            ("Mean clinical precision@k", f"{summary['mean_clinical_precision_at_k']:.3f}"),
            ("Mean injury sim@k", f"{summary['mean_injury_similarity_at_k']:.3f}"),
            ("Mean min injury sim@k", f"{summary['mean_min_injury_similarity_at_k']:.3f}"),
            ("Mean loss sim@k", f"{summary['mean_loss_similarity_at_k']:.3f}"),
            ("Mean PSLA-band NDCG@k", f"{summary['mean_ndcg_at_k']:.3f}"),
            ("Random NDCG@k", f"{summary['mean_random_ndcg_at_k']:.3f}"),
            ("Mean precision@k", f"{summary['mean_precision_at_k']:.3f}"),
            ("Mean recall@k", f"{summary['mean_recall_at_k']:.3f}"),
            ("Within-tolerance rate", f"{summary['mean_within_tolerance_rate']:.3f}"),
            ("Median PSLA MAE", _fmt_money(summary["median_psla_mae"])),
            ("Median log-ratio error", f"{summary['median_psla_median_log_ratio']:.3f}"),
            (
                "Median Spearman(sim, PSLA)",
                f"{summary['median_spearman_similarity_psla']:.3f}",
            ),
            ("Mean injury overlap@k", f"{summary['mean_injury_overlap_at_k']:.3f}"),
            ("Mean loss overlap@k", f"{summary['mean_loss_overlap_at_k']:.3f}"),
        ]
        for label, value in rows:
            table.add_row(label, value)
        console.print(table)

    # -- persistence ------------------------------------------------------
    @override
    def save_results(self) -> None:
        if self._result is None:
            raise RuntimeError("run() must be called before save_results()")

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self._result, f, indent=2, ensure_ascii=False, default=str)

        if self._per_query:
            csv_path = output_dir / "per_query.csv"
            fields = list(QueryEvaluation.__dataclass_fields__.keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for evaluation in self._per_query:
                    writer.writerow(evaluation.to_dict())
            console.print(f"[bold green]Per-query metrics: {csv_path}[/bold green]")

        benchmark = self._result.get("benchmark")
        tuning = benchmark.get("tuning") if isinstance(benchmark, dict) else None
        if tuning:
            weights_path = output_dir / "tuned_weights.json"
            with weights_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "weights": tuning["best_weights"],
                        "objective": tuning["objective"],
                        "clinical_alpha": self._config.clinical_alpha,
                        "min_injury_similarity": self._config.min_injury_similarity,
                        "test_clinical_ndcg_at_k": tuning["test_best"][
                            "mean_clinical_ndcg_at_k"
                        ],
                        "test_default_clinical_ndcg_at_k": tuning["test_default"][
                            "mean_clinical_ndcg_at_k"
                        ],
                        "test_ndcg_at_k": tuning["test_best"]["mean_ndcg_at_k"],
                        "test_default_ndcg_at_k": tuning["test_default"][
                            "mean_ndcg_at_k"
                        ],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            console.print(
                f"[bold green]Tuned weights: {weights_path}[/bold green]"
            )

        console.print(
            f"\n[bold green]Results saved to {metrics_path}[/bold green]"
        )


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"HK${value:,.0f}"
