from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.float_desc import FloatDesc


class FloatAttribute(SimpleAttribute):
    """Python has no distinct float32 type; behaves like DoubleAttribute."""

    def __init__(self, desc: FloatDesc, value: float):
        super().__init__(desc)
        self.value = value

    @override
    def get_value_as_string(self) -> str:
        return str(self.value)

    def __lt__(self, other: FloatAttribute) -> bool:
        return self.value < other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FloatAttribute):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)
