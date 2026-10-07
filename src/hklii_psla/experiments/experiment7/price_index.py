"""Hong Kong price-index deflation for PSLA awards.

PSLA awards in the corpus span 1969–2020.  Nominal HKD amounts are not directly
comparable across that range, so every award is converted to a common base year
(default 2020, the latest year in ``data/price_index.csv``) before any
closeness or band metric is computed.

The deflator is::

    real(amount, year) = amount * index[base_year] / index[year]

i.e. an old award is scaled *up* to what it would be worth in the base year.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRICE_INDEX_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "price_index.csv"
)


@dataclass(frozen=True, slots=True)
class PriceIndex:
    """A year -> index-value table with a chosen base year."""

    index: dict[int, float]
    base_year: int
    source: str = ""

    # -- construction -----------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        path: str | Path = DEFAULT_PRICE_INDEX_PATH,
        base_year: int | None = None,
    ) -> PriceIndex:
        rows: dict[int, float] = {}
        with Path(path).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                year_raw = (row.get("Year") or "").strip()
                value_raw = (row.get("Price Index") or "").strip()
                if not year_raw or not value_raw:
                    continue
                rows[int(year_raw)] = float(value_raw)
        if not rows:
            raise ValueError(f"Price index file has no usable rows: {path}")
        resolved_base = base_year if base_year is not None else max(rows)
        if resolved_base not in rows:
            raise ValueError(
                f"Base year {resolved_base} not present in price index {path}"
            )
        return cls(index=rows, base_year=resolved_base, source=str(path))

    # -- queries ----------------------------------------------------------
    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted(self.index))

    def has_year(self, year: int | None) -> bool:
        return year is not None and year in self.index

    def deflator(self, year: int) -> float:
        """Multiplier converting a nominal amount in ``year`` to the base year."""
        if year not in self.index:
            raise KeyError(f"Year {year} not in price index")
        return self.index[self.base_year] / self.index[year]

    def to_real(self, amount: float | None, year: int | None) -> float | None:
        """Inflation-adjust ``amount`` from ``year`` to the base year.

        Returns ``None`` when either the amount or the year is unavailable or
        the year is outside the index.  A non-positive amount is passed through
        unchanged (there is nothing to deflate).
        """
        if amount is None or year is None:
            return None
        if year not in self.index:
            return None
        if amount <= 0:
            return float(amount)
        return float(amount) * self.deflator(year)

    def to_nominal(
        self, amount_real: float | None, year: int | None
    ) -> float | None:
        """Inverse of :meth:`to_real`."""
        if amount_real is None or year is None or year not in self.index:
            return None
        if amount_real <= 0:
            return float(amount_real)
        return float(amount_real) / self.deflator(year)

    def __len__(self) -> int:
        return len(self.index)
