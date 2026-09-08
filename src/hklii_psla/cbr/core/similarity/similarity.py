from __future__ import annotations

from typing import ClassVar, override


class Similarity:
    """Immutable, cached similarity value in [0.0, 1.0], or -1.0 for invalid."""

    _sims: ClassVar[dict[float, Similarity]] = {}
    INVALID_SIM: ClassVar[Similarity]

    def __init__(self, d: float):
        self._value: float = d
        self._rounded_value: float = round(d * 100) / 100
        if self._rounded_value > 1 or (self._rounded_value < 0 and self._rounded_value != -1.00):
            self._rounded_value = -1.00

    @classmethod
    def get(cls, d: float) -> Similarity:
        cached = cls._sims.get(d)
        if cached is not None:
            return cached
        if (0.0 <= d <= 1.0) or d == -1.0:
            instance = cls(d)
            cls._sims[d] = instance
            return instance
        return cls.get(-1.0)

    @property
    def value(self) -> float:
        return self._value

    @property
    def rounded_value(self) -> float:
        return self._rounded_value

    def to_string(self) -> str:
        if self._value > 1 or self._value < 0:
            return "NaN"
        return str(self._value)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Similarity):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"Similarity({self._value})"


Similarity.INVALID_SIM = Similarity.get(-1.00)
