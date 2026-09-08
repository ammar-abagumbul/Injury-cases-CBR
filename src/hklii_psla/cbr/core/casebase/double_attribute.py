from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.double_desc import DoubleDesc


class DoubleAttribute(SimpleAttribute):

    def __init__(self, desc: DoubleDesc, value: float):
        super().__init__(desc)
        self.value = value

    @override
    def get_value_as_string(self) -> str:
        return str(self.value)

    def __lt__(self, other: DoubleAttribute) -> bool:
        return self.value < other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DoubleAttribute):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)
