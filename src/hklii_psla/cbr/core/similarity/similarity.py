from __future__ import annotations

from typing import ClassVar, override


class Similarity:
    _sims: ClassVar[dict[float, Similarity]] = {}
    INVALID_SIM: ClassVar[Similarity | None] = None

    def __init__(self, d: float):
        self._value: float = d
        self._rounded_value: float = -1.00
        self.update_rounded_value()

    @classmethod
    def get(cls, d: float) -> Similarity:
        if d in cls._sims:
            return cls._sims[d]

        if (0.0 <= d <= 1.0) or d == -1.0:
            instance = cls(d)
            cls._sims[d] = instance
            return instance

        return cls.get(-1.0)

    def update_rounded_value(self):
        rv = round(self._rounded_value * 100)  / 100
        if rv > 1 or (rv < 0 and rv != -1.00):
           self._rounded_value = -1.00

    @property
    def rounded_value(self):
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

Similarity.INVALID_SIM = Similarity.get(-1.00)
