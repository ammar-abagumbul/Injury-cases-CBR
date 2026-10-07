"""Experiment 7 evaluation metrics.

Two complementary views of "did we retrieve the right cases?" are computed:

* **PSLA closeness** — how near the retrieved awards are to the query's award
  (absolute error, median log-ratio error, within-tolerance rate) and how
  strongly the retrieval similarity correlates with PSLA proximity (Spearman).
* **PSLA-band relevance (IR)** — treating cases within a factor
  ``1 + tolerance`` of the query award as relevant, the usual precision@k,
  recall@k and graded NDCG@k, alongside random baselines.

Feature overlap (ICD-11 injury similarity and loss-category overlap) is also
reported so we can confirm the retriever still finds *clinically* similar
cases, not just financially similar ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hklii_psla.experiments.experiment4.case_loader import (
    exact_match,
    max_overlap_similarity,
)
from hklii_psla.experiments.experiment4.icd_similarity import IcdTaxonomy
from hklii_psla.experiments.experiment7.features import (
    CorpusCase,
    graded_relevance,
    is_relevant,
    log_ratio_error,
    relative_error,
)


@dataclass(slots=True)
class ClinicalRelevance:
    """Per-candidate clinical relevance derived from the injury/loss similarities.

    ``gains`` is a graded clinical relevance in ``[0, 1]`` (a weighted blend of
    the graded ICD injury similarity and the loss-set overlap); ``relevant``
    marks gains at or above the calibrated threshold.  ``injury``/``loss`` are
    the masked local similarities (0 where unknown) with their presence masks.
    """

    gains: np.ndarray
    relevant: np.ndarray
    injury: np.ndarray
    injury_known: np.ndarray
    loss: np.ndarray
    loss_known: np.ndarray

    def select(self, index: int | np.ndarray) -> ClinicalRelevance:
        """A row slice, aligned with one query's candidates."""
        return ClinicalRelevance(
            gains=self.gains[index],
            relevant=self.relevant[index],
            injury=self.injury[index],
            injury_known=self.injury_known[index],
            loss=self.loss[index],
            loss_known=self.loss_known[index],
        )


def clinical_relevance(
    S: np.ndarray,
    known: np.ndarray,
    attribute_names: Sequence[str],
    *,
    threshold: float = 0.15,
    injury_weight: float = 0.75,
) -> ClinicalRelevance:
    """Derive the clinical relevance label from per-attribute similarities.

    Uses a deliberately **loose** criterion: graded ICD injury similarity,
    optionally reinforced by shared loss categories.  It never demands an exact
    code match, because the extraction cannot be trusted to assign one.
    """
    S = np.asarray(S, dtype=np.float64)
    known = np.asarray(known, dtype=bool)
    names = list(attribute_names)

    def column(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name in names:
            i = names.index(name)
            return S[..., i], known[..., i]
        zeros = np.zeros(S.shape[:-1], dtype=np.float64)
        return zeros, np.zeros(S.shape[:-1], dtype=bool)

    injury, injury_known = column("injuries")
    loss, loss_known = column("losses")
    injury = np.where(injury_known, injury, 0.0)
    loss = np.where(loss_known, loss, 0.0)

    if "losses" in names and injury_weight < 1.0:
        gains = injury_weight * injury + (1.0 - injury_weight) * loss
    else:
        gains = injury
    gains = np.clip(gains, 0.0, 1.0)
    relevant = gains >= threshold
    return ClinicalRelevance(
        gains=gains,
        relevant=relevant,
        injury=injury,
        injury_known=injury_known,
        loss=loss,
        loss_known=loss_known,
    )


@dataclass(slots=True)
class QueryEvaluation:
    """Per-query metrics for one leave-one-out retrieval."""

    query_id: str
    k: int
    n_candidates: int
    n_relevant: int
    # PSLA closeness over the retrieved top-k
    psla_mae: float | None
    psla_median_abs_error: float | None
    psla_mape: float | None
    psla_median_log_ratio: float | None
    within_tolerance_rate: float
    # IR / band (PSLA-band relevance)
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    random_precision_at_k: float
    random_recall_at_k: float
    random_ndcg_at_k: float
    # Clinical relevance (injury/loss), independent of PSLA
    clinical_precision_at_k: float
    clinical_recall_at_k: float
    clinical_ndcg_at_k: float
    injury_similarity_at_k: float
    min_injury_similarity_at_k: float
    loss_similarity_at_k: float
    # feature overlap
    injury_overlap_at_k: float
    loss_overlap_at_k: float
    # alignment
    spearman_similarity_psla: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Element similarity helpers
# ---------------------------------------------------------------------------
def make_icd_element_sim(taxonomy: IcdTaxonomy):
    """``s(a, b)`` for two ICD codes via the taxonomy's normalised LCA depth."""

    def sim(a: str, b: str) -> float:
        if a == b:
            return 1.0
        if taxonomy.is_known(a) and taxonomy.is_known(b):
            return taxonomy.similarity(a, b)
        return 0.0

    return sim


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------
def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Average ranks (1 = largest), ties shared — mirrors Experiment 4."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx, my = rx.mean(), ry.mean()
    vx = float(((rx - mx) ** 2).sum())
    vy = float(((ry - my) ** 2).sum())
    if vx == 0.0 or vy == 0.0:
        return 0.0
    cov = float(((rx - mx) * (ry - my)).sum())
    return cov / math.sqrt(vx * vy)


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------
def evaluate_query(
    query: CorpusCase,
    candidates: Sequence[CorpusCase],
    similarities: Sequence[float],
    *,
    k: int,
    tolerance: float,
    taxonomy: IcdTaxonomy,
    mask: Sequence[bool] | None = None,
    clinical: ClinicalRelevance | None = None,
) -> QueryEvaluation:
    """Evaluate one retrieval.

    ``candidates`` is aligned with ``similarities``.  ``mask`` marks the valid
    candidates (leave-one-out sets the query's own entry to ``False``); invalid
    entries are ignored in every metric.  Candidates without a real PSLA are
    treated as irrelevant / zero-gain.
    """
    n = len(candidates)
    sims = np.asarray(similarities, dtype=np.float64)
    valid = (
        np.ones(n, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    n_valid = int(valid.sum())
    query_real = query.psla_real

    gains = np.asarray(
        [
            graded_relevance(query_real, c.psla_real, tolerance)
            for c in candidates
        ],
        dtype=np.float64,
    )
    relevant = np.asarray(
        [is_relevant(query_real, c.psla_real, tolerance) for c in candidates],
        dtype=bool,
    )
    gains = np.where(valid, gains, 0.0)
    relevant = np.where(valid, relevant, False)
    sims = np.where(valid, sims, -np.inf)
    n_relevant = int(relevant.sum())

    kk = min(k, n_valid)
    top = np.argsort(-sims, kind="stable")[:kk]
    top_gains = gains[top]
    top_relevant = relevant[top]

    # -- IR -----------------------------------------------------------------
    precision = float(top_relevant.mean()) if kk else 0.0
    recall = float(top_relevant.sum() / n_relevant) if n_relevant else 0.0

    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = float((top_gains * discounts).sum())
    ideal_gains = np.sort(gains)[::-1][:kk]
    idcg = float((ideal_gains * discounts).sum())
    ndcg = dcg / idcg if idcg > 0 else 0.0

    random_precision = n_relevant / n_valid if n_valid else 0.0
    random_recall = kk / n_valid if n_valid else 0.0
    expected_dcg = (
        float(gains.sum() / n_valid * discounts.sum()) if n_valid else 0.0
    )
    random_ndcg = expected_dcg / idcg if idcg > 0 else 0.0

    # -- PSLA closeness -----------------------------------------------------
    top_cases = [candidates[i] for i in top]
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    log_errors: list[float] = []
    for c in top_cases:
        e = log_ratio_error(query_real, c.psla_real)
        if e is None:
            continue
        log_errors.append(e)
        if c.psla_real is not None and query_real is not None:
            abs_errors.append(abs(c.psla_real - query_real))
        re = relative_error(query_real, c.psla_real)
        if re is not None:
            rel_errors.append(re)

    within = float(top_relevant.mean()) if kk else 0.0
    psla_mae = float(np.mean(abs_errors)) if abs_errors else None
    psla_median = float(np.median(abs_errors)) if abs_errors else None
    psla_mape = float(np.mean(rel_errors)) if rel_errors else None
    psla_median_log = float(np.median(log_errors)) if log_errors else None

    # -- feature overlap ----------------------------------------------------
    icd_sim = make_icd_element_sim(taxonomy)
    injury_overlap = float(
        np.mean(
            [
                max_overlap_similarity(query.injuries, c.injuries, icd_sim)
                for c in top_cases
            ]
        )
    ) if top_cases else 0.0
    loss_overlap = float(
        np.mean(
            [
                max_overlap_similarity(query.losses, c.losses, exact_match)
                for c in top_cases
            ]
        )
    ) if top_cases else 0.0

    # -- alignment ----------------------------------------------------------
    rho = spearman(
        sims[valid].tolist(),
        gains[valid].tolist(),
    )

    # -- clinical relevance (injury/loss), independent of PSLA --------------
    if clinical is not None:
        c_gains = np.where(valid, clinical.gains, 0.0)
        c_rel = np.where(valid, clinical.relevant, False)
        n_clinical = int(c_rel.sum())
        top_c_gains = c_gains[top]
        top_c_rel = c_rel[top]
        clinical_precision = float(top_c_rel.mean()) if kk else 0.0
        clinical_recall = (
            float(top_c_rel.sum() / n_clinical) if n_clinical else 0.0
        )
        c_dcg = float((top_c_gains * discounts).sum())
        ideal_c = np.sort(c_gains)[::-1][:kk]
        c_idcg = float((ideal_c * discounts).sum())
        clinical_ndcg = c_dcg / c_idcg if c_idcg > 0 else 0.0
        top_injury = clinical.injury[top]
        top_loss = clinical.loss[top]
        mean_injury = float(top_injury.mean()) if kk else 0.0
        min_injury = float(top_injury.min()) if kk else 0.0
        mean_loss = float(top_loss.mean()) if kk else 0.0
    else:
        clinical_precision = clinical_recall = clinical_ndcg = 0.0
        mean_injury = min_injury = mean_loss = 0.0

    return QueryEvaluation(
        query_id=query.case_id,
        k=kk,
        n_candidates=n_valid,
        n_relevant=n_relevant,
        psla_mae=psla_mae,
        psla_median_abs_error=psla_median,
        psla_mape=psla_mape,
        psla_median_log_ratio=psla_median_log,
        within_tolerance_rate=within,
        precision_at_k=precision,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        random_precision_at_k=random_precision,
        random_recall_at_k=random_recall,
        random_ndcg_at_k=random_ndcg,
        clinical_precision_at_k=clinical_precision,
        clinical_recall_at_k=clinical_recall,
        clinical_ndcg_at_k=clinical_ndcg,
        injury_similarity_at_k=mean_injury,
        min_injury_similarity_at_k=min_injury,
        loss_similarity_at_k=mean_loss,
        injury_overlap_at_k=injury_overlap,
        loss_overlap_at_k=loss_overlap,
        spearman_similarity_psla=rho,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics for weight tuning (no per-query Python loops)
# ---------------------------------------------------------------------------
def weighted_global_similarities(
    S: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``(Q, C, A) @ (A,)`` normalised to ``[0, 1]``."""
    denom = float(weights.sum())
    if denom <= 0:
        return np.zeros(S.shape[:2], dtype=np.float64)
    return (S.astype(np.float64) @ weights.astype(np.float64)) / denom


def aggregate_metrics(
    G: np.ndarray,
    gains: np.ndarray,
    relevant: np.ndarray,
    *,
    k: int,
    clinical: ClinicalRelevance | None = None,
) -> dict[str, float]:
    """Vectorised mean IR/PSLA metrics over many queries.

    ``G`` is ``(Q, C)`` masked global similarity (invalid candidates must be
    ``-inf``); ``gains`` and ``relevant`` are aligned ``(Q, C)`` arrays.  When
    ``clinical`` is supplied, the clinical-relevance metrics are reported
    alongside the PSLA-band ones.
    """
    q, c = G.shape
    kk = min(k, c)
    top = np.argsort(-G, axis=1, kind="stable")[:, :kk]
    rows = np.arange(q)[:, None]

    top_relevant = relevant[rows, top]
    precision = top_relevant.mean(axis=1)
    n_relevant = relevant.sum(axis=1)
    recall = np.divide(
        top_relevant.sum(axis=1),
        n_relevant,
        out=np.zeros(q),
        where=n_relevant > 0,
    )

    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    top_gains = gains[rows, top]
    dcg = (top_gains * discounts).sum(axis=1)
    ideal = np.sort(gains, axis=1)[:, ::-1][:, :kk]
    idcg = (ideal * discounts).sum(axis=1)
    ndcg = np.divide(dcg, idcg, out=np.zeros(q), where=idcg > 0)

    random_precision = n_relevant / c
    random_recall = np.full(q, kk / c)
    expected_dcg = gains.sum(axis=1) / c * discounts.sum()
    random_ndcg = np.divide(
        expected_dcg, idcg, out=np.zeros(q), where=idcg > 0
    )

    out = {
        "mean_precision_at_k": float(precision.mean()),
        "mean_recall_at_k": float(recall.mean()),
        "mean_ndcg_at_k": float(ndcg.mean()),
        "mean_within_tolerance_rate": float(top_relevant.mean()),
        "mean_random_precision_at_k": float(random_precision.mean()),
        "mean_random_recall_at_k": float(random_recall.mean()),
        "mean_random_ndcg_at_k": float(random_ndcg.mean()),
        "mean_n_relevant": float(n_relevant.mean()),
    }

    if clinical is not None:
        c_gains = clinical.gains
        c_rel = clinical.relevant
        top_c_rel = c_rel[rows, top]
        n_clinical = c_rel.sum(axis=1)
        c_precision = top_c_rel.mean(axis=1)
        c_recall = np.divide(
            top_c_rel.sum(axis=1),
            n_clinical,
            out=np.zeros(q),
            where=n_clinical > 0,
        )
        c_dcg = (c_gains[rows, top] * discounts).sum(axis=1)
        c_ideal = np.sort(c_gains, axis=1)[:, ::-1][:, :kk]
        c_idcg = (c_ideal * discounts).sum(axis=1)
        c_ndcg = np.divide(c_dcg, c_idcg, out=np.zeros(q), where=c_idcg > 0)
        top_injury = clinical.injury[rows, top]
        top_loss = clinical.loss[rows, top]
        out.update(
            {
                "mean_clinical_precision_at_k": float(c_precision.mean()),
                "mean_clinical_recall_at_k": float(c_recall.mean()),
                "mean_clinical_ndcg_at_k": float(c_ndcg.mean()),
                "mean_injury_similarity_at_k": float(top_injury.mean()),
                "mean_min_injury_similarity_at_k": float(top_injury.min(axis=1).mean()),
                "mean_loss_similarity_at_k": float(top_loss.mean()),
                "mean_n_clinical_relevant": float(n_clinical.mean()),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Aggregation over QueryEvaluation objects
# ---------------------------------------------------------------------------
_MEAN_FIELDS = (
    "within_tolerance_rate",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "random_precision_at_k",
    "random_recall_at_k",
    "random_ndcg_at_k",
    "clinical_precision_at_k",
    "clinical_recall_at_k",
    "clinical_ndcg_at_k",
    "injury_similarity_at_k",
    "min_injury_similarity_at_k",
    "loss_similarity_at_k",
    "injury_overlap_at_k",
    "loss_overlap_at_k",
)
_MEDIAN_FIELDS = (
    "psla_mae",
    "psla_median_abs_error",
    "psla_mape",
    "psla_median_log_ratio",
    "spearman_similarity_psla",
)


def summarise_evaluations(
    evaluations: Sequence[QueryEvaluation],
) -> dict[str, Any]:
    if not evaluations:
        return {"n_queries": 0}
    summary: dict[str, Any] = {"n_queries": len(evaluations)}
    for field in _MEAN_FIELDS:
        values = [
            getattr(e, field) for e in evaluations if getattr(e, field) is not None
        ]
        summary[f"mean_{field}"] = float(np.mean(values)) if values else None
    for field in _MEDIAN_FIELDS:
        values = [
            getattr(e, field) for e in evaluations if getattr(e, field) is not None
        ]
        summary[f"median_{field}"] = float(np.median(values)) if values else None
    summary["mean_n_relevant"] = float(
        np.mean([e.n_relevant for e in evaluations])
    )
    return summary
