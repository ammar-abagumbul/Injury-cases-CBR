"""Retrieval result types and a thin retriever around :class:`CorpusCBRModel`.

The retriever turns raw ``[(case_id, similarity)]`` pairs into
:class:`RankedCase` records carrying the candidate's nominal and real PSLA, so
callers (and the evaluation layer) can reason about compensation closeness.
Optional PSLA-band filtering restricts the candidate pool to a real-HKD window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from hklii_psla.experiments.experiment7.cbr_model import CorpusCBRModel
from hklii_psla.experiments.experiment7.features import CorpusCase


@dataclass(frozen=True, slots=True)
class RankedCase:
    case_id: str
    similarity: float
    psla_nominal: float | None
    psla_real: float | None
    year: int | None
    extracted_severity: str | None = None
    award_severity: str | None = None

    @property
    def has_psla(self) -> bool:
        return self.psla_real is not None and self.psla_real > 0


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query_id: str
    k: int
    weights: dict[str, float]
    psla_band: tuple[float, float] | None
    results: list[RankedCase] = field(default_factory=list)

    @property
    def case_ids(self) -> list[str]:
        return [r.case_id for r in self.results]

    @property
    def similarities(self) -> list[float]:
        return [r.similarity for r in self.results]

    @property
    def reals(self) -> list[float | None]:
        return [r.psla_real for r in self.results]


class CBRFeatureRetriever:
    """Retrieve similar corpus cases for a feature query."""

    def __init__(self, model: CorpusCBRModel) -> None:
        self.model = model
        self._by_id: dict[str, CorpusCase] = {c.case_id: c for c in model.cases}

    def candidate_ids(
        self,
        psla_band: tuple[float, float] | None = None,
        require_psla: bool = False,
    ) -> list[str]:
        if psla_band is None and not require_psla:
            return self.model.candidate_ids
        out: list[str] = []
        for c in self.model.cases:
            if require_psla and not c.has_psla:
                continue
            if psla_band is not None and (
                c.psla_real is None
                or not (psla_band[0] <= c.psla_real <= psla_band[1])
            ):
                continue
            out.append(c.case_id)
        return out

    def retrieve(
        self,
        query: CorpusCase,
        k: int = 10,
        weights: dict[str, float] | None = None,
        exclude: str | None = None,
        psla_band: tuple[float, float] | None = None,
        candidate_ids: Sequence[str] | None = None,
        require_psla: bool = False,
        *,
        clinical_alpha: float = 0.7,
        min_injury_similarity: float | None = None,
    ) -> RetrievalResult:
        allowed = (
            list(candidate_ids)
            if candidate_ids is not None
            else self.candidate_ids(psla_band, require_psla=require_psla)
        )
        ranked = self.model.retrieve(
            query,
            k=k,
            weights=weights,
            exclude=exclude,
            candidate_ids=allowed,
            clinical_alpha=clinical_alpha,
            min_injury_similarity=min_injury_similarity,
        )
        results: list[RankedCase] = []
        for case_id, similarity in ranked:
            case = self._by_id.get(case_id)
            results.append(
                RankedCase(
                    case_id=case_id,
                    similarity=similarity,
                    psla_nominal=case.psla_nominal if case else None,
                    psla_real=case.psla_real if case else None,
                    year=case.year if case else None,
                    extracted_severity=case.overall_category if case else None,
                    award_severity=case.award_severity if case else None,
                )
            )
        merged_weights = dict(self.model._weights)
        if weights:
            merged_weights.update({k2: float(v) for k2, v in weights.items()})
        return RetrievalResult(
            query_id=query.case_id,
            k=k,
            weights=merged_weights,
            psla_band=psla_band,
            results=results,
        )
