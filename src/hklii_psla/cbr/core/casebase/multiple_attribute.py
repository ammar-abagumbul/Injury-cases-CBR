from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.attribute import Attribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc


class MultipleAttribute(Attribute):
    """A set of "simple" values for one attribute description. Do not put
    SpecialAttribute values in here."""

    def __init__(self, desc: AttributeDesc, values: list[Attribute]):
        self.attribute_desc = desc
        self.values = values

    def add_value(self, value: Attribute) -> None:
        from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if isinstance(value, SpecialAttribute):
            return
        if isinstance(value, SimpleAttribute) and value.attribute_desc != self.attribute_desc:
            return
        if value not in self.values:
            self.values.append(value)

    def remove_value(self, value: Attribute) -> None:
        if value in self.values:
            self.values.remove(value)

    @override
    def get_value_as_string(self) -> str:
        return ";".join(str(v) for v in self.values)

    def __str__(self) -> str:
        return self.get_value_as_string()
