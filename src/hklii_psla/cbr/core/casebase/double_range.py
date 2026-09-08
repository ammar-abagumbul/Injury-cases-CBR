from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.double_attribute import DoubleAttribute
from hklii_psla.cbr.core.casebase.range import Range
from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.double_desc import DoubleDesc
    from hklii_psla.cbr.core.project import Project


class DoubleRange(Range):

    def __init__(self, prj: Project, double_desc: DoubleDesc):
        super().__init__(prj)
        self._desc = double_desc
        self._double_atts: dict[float, DoubleAttribute] = {}

    @property
    def desc(self) -> DoubleDesc:
        return self._desc

    def contains_double(self, value: float) -> bool:
        return value in self._double_atts

    def get_double_value(self, value: float) -> DoubleAttribute | None:
        att = self._double_atts.get(value)
        if att is None and self._desc.min <= value <= self._desc.max:
            att = DoubleAttribute(self._desc, value)
            self._double_atts[value] = att
        return att

    def get_doubles(self):
        return self._double_atts.values()

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        assert self.project is not None
        if self.project.is_special_attribute(str(obj)):
            return self.project.get_special_attribute(str(obj))
        if isinstance(obj, (int, float)):
            return self.get_double_value(float(obj))
        return self.parse_value(str(obj))

    def get_value(self, value: float) -> SimpleAttribute | None:
        return self.get_double_value(value)

    def clean_data(self) -> None:
        for v in [
            v for v in self._double_atts
            if v < self._desc.min or v > self._desc.max
        ]:
            del self._double_atts[v]

    @override
    def parse_value(self, string: str):
        return self.get_double_value(float(string))
