from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.integer_attribute import IntegerAttribute
from hklii_psla.cbr.core.casebase.range import Range
from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.integer_desc import IntegerDesc
    from hklii_psla.cbr.core.project import Project


class IntegerRange(Range):

    def __init__(self, prj: Project, d: IntegerDesc):
        super().__init__(prj)
        self._desc = d
        self._int_atts: dict[int, IntegerAttribute] = {}

    @property
    def desc(self) -> IntegerDesc:
        return self._desc

    def contains_integer(self, value: int) -> bool:
        return value in self._int_atts

    def get_integer_value(self, value: int) -> IntegerAttribute | None:
        att = self._int_atts.get(value)
        if att is None and self._desc.min <= value <= self._desc.max:
            att = IntegerAttribute(self._desc, value)
            self._int_atts[value] = att
        return att

    def get_integers(self) -> dict[int, IntegerAttribute]:
        return self._int_atts

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        assert self.project is not None
        if self.project.is_special_attribute(str(obj)):
            return self.project.get_special_attribute(str(obj))
        if isinstance(obj, int):
            return self.get_integer_value(obj)
        return self.parse_value(str(obj))

    def get_value(self, v: int) -> SimpleAttribute | None:
        return self.get_integer_value(v)

    def clean_data(self) -> None:
        for v in [
            v for v in self._int_atts
            if v < self._desc.min or v > self._desc.max
        ]:
            del self._int_atts[v]

    @override
    def parse_value(self, string: str):
        return self.get_integer_value(int(string))
