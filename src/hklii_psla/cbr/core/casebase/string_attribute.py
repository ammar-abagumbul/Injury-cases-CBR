from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.string_desc import StringDesc


class StringAttribute(SimpleAttribute):

    def __init__(self, desc: StringDesc, v: str):
        super().__init__(desc)
        self.value = v

    @override
    def get_value_as_string(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StringAttribute):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)
