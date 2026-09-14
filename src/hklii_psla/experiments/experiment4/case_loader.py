from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A per-element similarity: s(a, b) -> [0.0, 1.0].
ElementSim = Callable[[str, str], float]

CSV_FIELDS: tuple[str, ...] = (
    "case_id",
    "gender",
    "age_at_accident",
    "age_at_trial",
    "injuries",
    "losses",
    "psla_amount",
)

DEFAULT_CASE_BASE_DIR = (
    Path(__file__).resolve().parents[3]
    / "output"
    / "experiments"
    / "reconciliation"
    / "batch_0"
)


@dataclass(frozen=True, slots=True)
class CaseFeatures:
    """A single case reduced to the features we retrieve on.

    ``injuries`` and ``losses`` are tuples so the record is hashable and can be
    treated as a set-valued feature.  ``as_tuple()`` exposes the plain tuple
    form ``(case_id, gender, age_at_accident, age_at_trial, injuries, losses,
    psla_amount)`` for callers that want to treat a case as a row.
    """

    case_id: str
    gender: str | None
    age_at_accident: int | None
    age_at_trial: int | None
    injuries: tuple[str, ...]
    losses: tuple[str, ...]
    psla_amount: float | None = None

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.case_id,
            self.gender,
            self.age_at_accident,
            self.age_at_trial,
            self.injuries,
            self.losses,
            self.psla_amount,
        )

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseFeatures:
        """Flatten one reconciliation JSON object into a :class:`CaseFeatures`."""
        metadata = data.get("metadata") or {}
        plaintiff = data.get("plaintiff") or {}
        injuries = (data.get("injuries") or {}).get("injuries") or []
        losses = (data.get("losses") or {}).get("losses") or []
        psla = data.get("psla") or {}

        return cls(
            case_id=_clean(metadata.get("neutral_citation"))
            or _clean(metadata.get("action_number"))
            or "",
            gender=_clean(plaintiff.get("gender")) or None,
            age_at_accident=_as_int(plaintiff.get("age_at_accident")),
            age_at_trial=_as_int(plaintiff.get("age_at_trial")),
            injuries=_dedupe(
                _clean(inj.get("icd_code")) for inj in injuries if isinstance(inj, dict)
            ),
            losses=_dedupe(
                _clean(loss.get("category")) for loss in losses if isinstance(loss, dict)
            ),
            psla_amount=_as_float(psla.get("amount")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> CaseFeatures:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # -- serialisation -----------------------------------------------------
    def to_row(self) -> dict[str, str]:
        """CSV-friendly row: lists become JSON strings, ``None`` becomes ``""``."""
        return {
            "case_id": self.case_id,
            "gender": self.gender or "",
            "age_at_accident": "" if self.age_at_accident is None else str(self.age_at_accident),
            "age_at_trial": "" if self.age_at_trial is None else str(self.age_at_trial),
            "injuries": json.dumps(list(self.injuries), ensure_ascii=False),
            "losses": json.dumps(list(self.losses), ensure_ascii=False),
            "psla_amount": "" if self.psla_amount is None else str(self.psla_amount),
        }

    @classmethod
    def from_row(cls, row: dict[str, str]) -> CaseFeatures:
        return cls(
            case_id=row.get("case_id", ""),
            gender=row.get("gender") or None,
            age_at_accident=_as_int(row.get("age_at_accident")),
            age_at_trial=_as_int(row.get("age_at_trial")),
            injuries=tuple(json.loads(row.get("injuries") or "[]")),
            losses=tuple(json.loads(row.get("losses") or "[]")),
            psla_amount=_as_float(row.get("psla_amount")),
        )


def load_case(path: str | Path) -> CaseFeatures:
    """Load a single reconciliation JSON file."""
    return CaseFeatures.from_json(path)


def load_case_base(paths: Iterable[str | Path]) -> list[CaseFeatures]:
    """Load an explicit collection of JSON files."""
    return [load_case(p) for p in paths]


def load_case_base_dir(
    directory: str | Path = DEFAULT_CASE_BASE_DIR,
    pattern: str = "*.json",
) -> list[CaseFeatures]:
    """Load every JSON file in ``directory`` (sorted by filename)."""
    return [load_case(p) for p in sorted(Path(directory).glob(pattern))]


def write_csv(records: Sequence[CaseFeatures], path: str | Path) -> None:
    """Write records to CSV; set-valued features are stored as JSON strings."""
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def read_csv(path: str | Path) -> list[CaseFeatures]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [CaseFeatures.from_row(row) for row in csv.DictReader(f)]


def max_overlap_similarity(
    a: Sequence[str],
    b: Sequence[str],
    s: ElementSim,
) -> float:
    """Symmetric maximum-overlap similarity of two sets, in ``[0.0, 1.0]``.

        Sim(A, B) = 1/2 * ( mean_{a in A} max_{b in B} s(a, b)
                          + mean_{b in B} max_{a in A} s(a, b) )

    Each element of one set is matched to its best partner in the other; the
    two directed averages are then averaged so the result is symmetric.  Two
    empty sets are treated as perfect agreement (1.0); one empty set scores 0.0.
    """
    a = list(a)
    b = list(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def directed(x: list[str], y: list[str]) -> float:
        return sum(max(s(xi, yj) for yj in y) for xi in x) / len(x)

    return 0.5 * (directed(a, b) + directed(b, a))


def exact_match(a: str, b: str) -> float:
    """1.0 iff the two strings are equal — the right default for loss categories."""
    return 1.0 if a == b else 0.0


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def token_jaccard(a: str, b: str) -> float:
    """Jaccard overlap of lower-cased word tokens — a default for injury text.

    Cheap, dependency-free, and order-insensitive.  Swap in a stronger metric
    (e.g. the ICD-11 taxonomy similarity from ``experiments/experiment4``) by
    passing a different ``s`` to :func:`max_overlap_similarity`.
    """
    ta = set(_TOKEN_RE.findall(a.lower()))
    tb = set(_TOKEN_RE.findall(b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def case_similarity(
    a: CaseFeatures,
    b: CaseFeatures,
    *,
    injury_sim: ElementSim = token_jaccard,
    loss_sim: ElementSim = token_jaccard,
) -> dict[str, float]:
    """Per-feature similarity of two cases for the set-valued features."""
    return {
        "injuries": max_overlap_similarity(a.injuries, b.injuries, injury_sim),
        "losses": max_overlap_similarity(a.losses, b.losses, loss_sim),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(value: Any) -> str:
    """Collapse whitespace; ``None`` -> ``""``."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    """Drop empties and duplicates while preserving first-seen order."""
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    cases = load_case_base_dir()
    print(f"loaded {len(cases)} cases from {DEFAULT_CASE_BASE_DIR}\n")
    for c in cases:
        print(f"{c.case_id:20} gender={c.gender!s:6} age_acc={c.age_at_accident!s:4} age_trial={c.age_at_trial!s:4} injuries={len(c.injuries)} losses={len(c.losses)} psla={c.psla_amount}")

    print("\npairwise set-valued similarity (injuries=token Jaccard, losses=exact):")
    for i, a in enumerate(cases):
        for b in cases[i + 1 :]:
            sim = case_similarity(a, b)
            print(f"  {a.case_id:20} vs {b.case_id:20} injuries={sim['injuries']:.3f} losses={sim['losses']:.3f}")


if __name__ == "__main__":
    main()
