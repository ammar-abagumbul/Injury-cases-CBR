from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.interval_desc import IntervalDesc


class IntervalAttribute(SimpleAttribute):

    def __init__(self, desc: IntervalDesc, interval: tuple[float, float]):
        super().__init__(desc)
        self.interval = interval

    @override
    def get_value_as_string(self) -> str:
        return str(self.interval)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalAttribute):
            return NotImplemented
        return self.attribute_desc is other.attribute_desc and self.interval == other.interval

    def __hash__(self) -> int:
        return hash(self.interval)

    def __str__(self) -> str:
        return str(self.interval)
