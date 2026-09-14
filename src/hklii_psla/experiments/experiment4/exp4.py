"""Experiment 4 — CBR case ranking.

Rank a set of candidate cases by similarity to a single query case (using the
migrated myCBR package), then score how well the predicted ranking matches a
human-specified "correct" order with Spearman rank correlation.

Inputs are always pre-extracted ``Case`` JSON files; loading is delegated to
``case_loader``.  See ``PLAN.md`` in this folder for the full design.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, override

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from hklii_psla.cbr.core.retrieval.sequential_retrieval import SequentialRetrieval
from hklii_psla.experiments.experiment import BaseExperiment
from hklii_psla.experiments.experiment4.case_loader import (
    CaseFeatures,
    load_case,
    load_case_base,
    load_case_base_dir,
    write_csv,
)
from hklii_psla.experiments.experiment4.cbr_model import build_model

console = Console()


class Exp4Config(BaseModel):
    id: str
    query_case_path: str
    candidate_cases_path: str | None = None
    candidate_case_paths: list[str] = []
    correct_order: list[str]
    output_dir: str
    weights: dict[str, float] | None = None


def _average_ranks(ids: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    """Ranks of ``ids`` ordered by ``values`` (descending), ties averaged."""
    order = sorted(range(len(ids)), key=lambda i: values[i], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[ids[order[k]]] = avg
        i = j + 1
    return ranks


def spearman_rank_correlation(
    predicted_ids: Sequence[str],
    predicted_sims: Sequence[float],
    correct_ids: Sequence[str],
) -> float:
    """Spearman ρ between a predicted ranking (ties averaged) and the correct order.

    ``predicted_ids`` / ``predicted_sims`` are aligned; ``correct_ids`` is the
    human order (best → worst).  Returns a value in ``[-1, 1]`` (1.0 = perfect).
    A degenerate ranking (all similarities tied) yields 0.0.
    """
    n = len(correct_ids)
    if n < 2:
        return 1.0

    pred_rank = _average_ranks(predicted_ids, predicted_sims)
    correct_rank = {cid: float(i + 1) for i, cid in enumerate(correct_ids)}

    xs = [pred_rank[cid] for cid in correct_ids]
    ys = [correct_rank[cid] for cid in correct_ids]

    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / math.sqrt(vx * vy)


class CBRRankingExperiment(BaseExperiment[Exp4Config], experiment_id="cbr-ranking"):

    _config: Exp4Config
    _config_model: ClassVar[type[BaseModel]] = Exp4Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)

        has_dir = self._config.candidate_cases_path is not None
        has_list = bool(self._config.candidate_case_paths)
        if has_dir == has_list:
            raise ValueError(
                "Exp4Config requires exactly one of 'candidate_cases_path' "
                + "(a directory of *.json) or 'candidate_case_paths' (an explicit "
                + "list of *.json); got "
                + f"candidate_cases_path={self._config.candidate_cases_path!r}, "
                + f"candidate_case_paths={self._config.candidate_case_paths!r}"
            )

        self._result: dict[str, Any] | None = None

    # -- pipeline ---------------------------------------------------------
    @override
    def run(self) -> None:
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        query = load_case(self._config.query_case_path)
        candidates = self._load_candidates()

        self._validate_candidates(candidates)

        csv_path = output_dir / "case_base.csv"
        write_csv([query, *candidates], csv_path)
        console.print(f"[cyan]Wrote flat features to {csv_path}[/cyan]")

        prj, _concept, cb, query_inst = build_model(query, candidates, self._config.weights)

        ranked = SequentialRetrieval(prj).retrieve_sorted(cb, query_inst)
        predicted = [inst.name for inst, _ in ranked]
        similarities = {inst.name: sim.value for inst, sim in ranked}

        correct = list(self._config.correct_order)
        rho = spearman_rank_correlation(
            predicted, [similarities[cid] for cid in predicted], correct
        )

        self._result = {
            "id": self._config.id,
            "query_case_id": query.case_id,
            "n_candidates": len(candidates),
            "correct_order": correct,
            "predicted_order": predicted,
            "similarities": similarities,
            "spearman_rho": rho,
            "weights": self._config.weights,
        }

        self._display(correct, predicted, similarities, rho)

    def _load_candidates(self) -> list[CaseFeatures]:
        if self._config.candidate_cases_path is not None:
            return load_case_base_dir(self._config.candidate_cases_path)
        return load_case_base(self._config.candidate_case_paths)

    def _validate_candidates(self, candidates: list[CaseFeatures]) -> None:
        if not candidates:
            raise RuntimeError("No candidate cases were loaded")

        ids = [c.case_id for c in candidates]
        if any(not cid for cid in ids):
            raise RuntimeError("Every candidate must have a non-empty case_id")

        duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
        if duplicates:
            raise RuntimeError(f"Duplicate candidate case_ids: {duplicates}")

        correct = list(self._config.correct_order)
        if len(set(correct)) != len(correct):
            raise RuntimeError("correct_order contains duplicate case_ids")

        missing = sorted(set(ids) - set(correct))
        unknown = sorted(set(correct) - set(ids))
        if missing or unknown:
            raise RuntimeError(
                "correct_order must list exactly the candidate case_ids.\n"
                + f"  missing from correct_order: {missing}\n"
                + f"  unknown in correct_order:   {unknown}\n"
                + f"  available candidate ids:    {sorted(ids)}"
            )

    def _display(
        self,
        correct: Sequence[str],
        predicted: Sequence[str],
        similarities: dict[str, float],
        rho: float,
    ) -> None:
        table = Table(title=f"Experiment 4 — CBR Case Ranking (Spearman ρ = {rho:.2f})")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Correct order", style="green")
        table.add_column("Predicted order", style="yellow")
        table.add_column("Similarity", justify="right", style="magenta")

        for i, cid in enumerate(correct):
            pred_id = predicted[i] if i < len(predicted) else ""
            sim = similarities.get(pred_id)
            table.add_row(
                str(i + 1),
                cid,
                pred_id,
                "" if sim is None else f"{sim:.3f}",
            )
        console.print(table)

    # -- persistence ------------------------------------------------------
    @override
    def save_results(self) -> None:
        if self._result is None:
            raise RuntimeError("run() must be called before save_results()")

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ranking.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._result, f, indent=2, ensure_ascii=False)

        console.print(f"\n[bold green]Results saved to {path}[/bold green]")
