"""Loader for WHO's ICHI "Simple Tabulation" flat export (``data/ICHI.txt``).

ICHI stem codes are ``TARGET.ACTION.MEANS`` (e.g. ``VA1.AA.ZZ``). Checked
against the real data (not assumed): the ACTION segment (2nd) is a shared,
reused vocabulary of ~120 codes independent of target — ``AA`` always means
"Assessment of X" whatever X is, ``KD`` is always "Transplantation", etc.
That's the axis this project actually needs: it's the severity-relevant
"what kind of deed was this" signal. TARGET (650 distinct values) is mostly
redundant with ``Injury.body_part`` we already extract, and MEANS (41
values, the technique/route) matters less for a severity proxy than ACTION
does. See ``ichi_actions.py`` for the hand-curated action -> severity tier
table built from this vocabulary.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ICHI_PATH = Path(__file__).resolve().parents[3] / "data" / "ICHI.txt"

_CODE_RE = re.compile(r"^([^.]+)\.([^.]+)\.([^.]+)$")


@dataclass(frozen=True)
class IchiEntity:
    code: str
    title: str
    class_kind: str  # "chapter" | "block" | "category"
    is_leaf: bool
    is_residual: bool


def split_code(code: str) -> tuple[str, str, str] | None:
    """Split a stem code into (target, action, means); None if malformed."""
    match = _CODE_RE.match(code)
    return match.groups() if match else None  # type: ignore[return-value]


class IchiTaxonomy:
    """``code -> entity`` index over ``data/ICHI.txt``."""

    def __init__(self, entities: dict[str, IchiEntity]):
        self._entities = entities

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_ICHI_PATH) -> IchiTaxonomy:
        entities: dict[str, IchiEntity] = {}
        with Path(path).open(encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # header
            for row in reader:
                if len(row) < 10:
                    continue
                code, title, class_kind = row[2], row[4], row[5]
                if not code:
                    continue
                is_residual = row[7].strip().lower() == "true"
                is_leaf = row[9].strip().lower() == "true"
                entities[code] = IchiEntity(
                    code=code,
                    title=title.lstrip("- ").strip(),
                    class_kind=class_kind,
                    is_leaf=is_leaf,
                    is_residual=is_residual,
                )
        return cls(entities)

    def __len__(self) -> int:
        return len(self._entities)

    def is_known(self, code: str) -> bool:
        return code in self._entities

    def describe(self, code: str) -> str | None:
        entity = self._entities.get(code)
        return entity.title if entity else None

    def get(self, code: str) -> IchiEntity | None:
        return self._entities.get(code)
