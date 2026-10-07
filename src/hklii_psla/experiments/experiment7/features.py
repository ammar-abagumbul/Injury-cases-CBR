"""Corpus features for Experiment 7.

A :class:`CorpusCase` is the flat, retrieval-ready view of one extracted
``Case`` JSON, augmented with the judgment year and the inflation-adjusted
("real") PSLA amount.  Loading lives here; the CBR model that consumes these
records lives in :mod:`hklii_psla.experiments.experiment7.cbr_model`.

The clinical features mirror Experiment 4 so the two experiments are directly
comparable, with treatment intensity and counts added as auxiliary signals:

* ``injuries``          — set of ICD-11 codes
* ``losses``            — set of canonical loss categories
* ``age_at_accident`` / ``age_at_trial`` — integers
* ``operations_count`` / ``hospitalisation_days`` / ``n_injuries`` / ``n_losses``
* ``gender``            — symbol (kept for completeness, off by default)
* ``overall_category``  — extracted severity label (off by default; the award
  band via :func:`severity_from_award` is authoritative)

``psla_nominal`` / ``psla_real`` are the *solution* side: the award never enters
retrieval similarity, it is only used to organise the corpus, to reconstruct the
severity band, and to evaluate retrieval.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hklii_psla.experiments.experiment7.price_index import PriceIndex

DEFAULT_CORPUS_DIR = (
    Path(__file__).resolve().parents[4]
    / "output"
    / "experiments"
    / "corpus_extraction"
    / "cases"
)

_YEAR_RE = re.compile(r"\[(\d{4})\]")
_ISO_DATE_RE = re.compile(r"(\d{4})")

# Severity is, for a decided case, the band of its year-adjusted award.  The
# boundaries are the published ``overall_category`` bands in real 2020 HKD and
# are half-open ``[lower, upper)``.  See ``notes.md`` §8.
SEVERITY_BANDS_2020: tuple[tuple[str, float, float], ...] = (
    ("Non-serious injury", 0.0, 564_000.0),
    ("Serious injury", 564_000.0, 761_000.0),
    ("Substantial injury", 761_000.0, 931_000.0),
    ("Gross disability", 931_000.0, 1_410_000.0),
    ("Disaster", 1_410_000.0, math.inf),
)


def severity_from_award(psla_real: float | None) -> str | None:
    """Reconstruct ``overall_category`` from a real (base-year) award.

    Severity is a deterministic function of the award, so for a decided case
    this recovers the category exactly; the extracted label is only used when no
    award is available.  Returns ``None`` when the award is missing or
    non-positive.
    """
    if psla_real is None or psla_real <= 0:
        return None
    for name, lower, upper in SEVERITY_BANDS_2020:
        if lower <= psla_real < upper:
            return name
    return None


@dataclass(frozen=True, slots=True)
class CorpusCase:
    """One corpus case reduced to the features Experiment 7 retrieves on."""

    case_id: str
    gender: str | None
    age_at_accident: int | None
    age_at_trial: int | None
    injuries: tuple[str, ...]
    losses: tuple[str, ...]
    psla_nominal: float | None = None
    year: int | None = None
    psla_real: float | None = None
    overall_category: str | None = None
    operations_count: int | None = None
    hospitalisation_days: int | None = None
    n_injuries: int = 0
    n_losses: int = 0

    # -- derived ----------------------------------------------------------
    @property
    def has_psla(self) -> bool:
        return self.psla_real is not None and self.psla_real > 0

    @property
    def award_severity(self) -> str | None:
        """Severity reconstructed from the award (authoritative for a decided case)."""
        return severity_from_award(self.psla_real)

    @property
    def effective_severity(self) -> str | None:
        """The award band when an award exists, else the extracted label."""
        return self.award_severity or self.overall_category

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.case_id,
            self.gender,
            self.age_at_accident,
            self.age_at_trial,
            self.injuries,
            self.losses,
            self.psla_nominal,
            self.year,
            self.psla_real,
        )

    # -- construction -----------------------------------------------------
    @classmethod
    def from_dict(
        cls, data: dict[str, Any], price_index: PriceIndex
    ) -> CorpusCase:
        metadata = data.get("metadata") or {}
        plaintiff = data.get("plaintiff") or {}
        injuries = (data.get("injuries") or {}).get("injuries") or []
        losses = (data.get("losses") or {}).get("losses") or []
        treatment = data.get("treatment") or {}
        psla = data.get("psla") or {}

        case_id = (
            _clean(metadata.get("neutral_citation"))
            or _clean(metadata.get("action_number"))
            or ""
        )
        year = extract_year(metadata)
        nominal = _as_float(psla.get("amount"))
        real = price_index.to_real(nominal, year)
        overall = _clean(
            (data.get("injuries") or {}).get("overall_category")
        ) or None

        return cls(
            case_id=case_id,
            gender=_clean(plaintiff.get("gender")) or None,
            age_at_accident=_as_int(plaintiff.get("age_at_accident")),
            age_at_trial=_as_int(plaintiff.get("age_at_trial")),
            injuries=_dedupe(
                _clean(inj.get("icd_code"))
                for inj in injuries
                if isinstance(inj, dict)
            ),
            losses=_dedupe(
                _clean(loss.get("category"))
                for loss in losses
                if isinstance(loss, dict)
            ),
            psla_nominal=nominal,
            year=year,
            psla_real=real,
            overall_category=overall,
            operations_count=_as_int(treatment.get("operations_count")),
            hospitalisation_days=_as_int(treatment.get("hospitalisation_days")),
            n_injuries=len(injuries),
            n_losses=len(losses),
        )

    @classmethod
    def from_json(cls, path: str | Path, price_index: PriceIndex) -> CorpusCase:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f), price_index)


def load_corpus(
    directory: str | Path = DEFAULT_CORPUS_DIR,
    price_index: PriceIndex | None = None,
    pattern: str = "*.json",
) -> list[CorpusCase]:
    """Load every ``*.json`` case in ``directory``, sorted by filename."""
    price_index = price_index or PriceIndex.from_csv()
    paths = sorted(Path(directory).glob(pattern))
    return [CorpusCase.from_json(p, price_index) for p in paths]


def extract_year(metadata: dict[str, Any]) -> int | None:
    """Best-effort judgment year: ``judgment_date`` first, then the citation."""
    date = metadata.get("judgment_date")
    if isinstance(date, str):
        match = _ISO_DATE_RE.search(date)
        if match:
            return int(match.group(1))
    citation = metadata.get("neutral_citation") or ""
    match = _YEAR_RE.search(str(citation))
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# PSLA helpers
# ---------------------------------------------------------------------------
def log_amount(amount: float | None) -> float | None:
    if amount is None or amount <= 0:
        return None
    return math.log(amount)


def log_ratio_error(a: float | None, b: float | None) -> float | None:
    """``|ln(a / b)|`` — a scale-free distance between two awards."""
    la, lb = log_amount(a), log_amount(b)
    if la is None or lb is None:
        return None
    return abs(la - lb)


def relative_error(query: float | None, candidate: float | None) -> float | None:
    """``|candidate - query| / query``."""
    if query is None or candidate is None or query <= 0:
        return None
    return abs(candidate - query) / query


def is_relevant(
    query_real: float | None, candidate_real: float | None, tolerance: float
) -> bool:
    """Relevant iff the awards are within a factor ``1 + tolerance``.

    ``tolerance=0.5`` means the candidate may be up to 50% above or below the
    query (ratio in ``[2/3, 1.5]``).
    """
    error = log_ratio_error(query_real, candidate_real)
    if error is None:
        return False
    return error <= math.log1p(tolerance)


def graded_relevance(
    query_real: float | None, candidate_real: float | None, tolerance: float
) -> float:
    """Linearly graded relevance in ``[0, 1]`` (1 at exact match, 0 at edge)."""
    error = log_ratio_error(query_real, candidate_real)
    if error is None:
        return 0.0
    edge = math.log1p(tolerance)
    if edge <= 0:
        return 0.0
    return max(0.0, 1.0 - error / edge)


def psla_band(amount: float | None, tolerance: float) -> int | None:
    """Integer band index for a real award (bands are multiplicative).

    Band ``k`` spans ``[base * (1 + tolerance) ** k, base * (1 + tolerance) ** (k + 1))``
    relative to the smallest positive award observed.  Provided so callers can
    organise the corpus into compensation bands; the evaluation itself uses the
    continuous :func:`graded_relevance`.
    """
    if amount is None or amount <= 0:
        return None
    return int(math.log(amount) / math.log1p(tolerance))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarise(cases: Sequence[CorpusCase]) -> dict[str, Any]:
    """Small descriptive summary used by the experiment banner."""
    with_psla = [c for c in cases if c.has_psla]
    reals = sorted(c.psla_real for c in with_psla if c.psla_real is not None)
    n = len(reals)
    return {
        "n_cases": len(cases),
        "n_with_psla": len(with_psla),
        "n_with_year": sum(1 for c in cases if c.year is not None),
        "median_real_psla": reals[n // 2] if n else None,
        "min_real_psla": reals[0] if n else None,
        "max_real_psla": reals[-1] if n else None,
    }
