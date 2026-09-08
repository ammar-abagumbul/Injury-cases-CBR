from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.float_attribute import FloatAttribute
from hklii_psla.cbr.core.casebase.range import Range
from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.float_desc import FloatDesc
    from hklii_psla.cbr.core.project import Project


class FloatRange(Range):

    def __init__(self, prj: Project, float_desc: FloatDesc):
        super().__init__(prj)
        self._desc = float_desc
        self._float_atts: dict[float, FloatAttribute] = {}

    @property
    def desc(self) -> FloatDesc:
        return self._desc

    def contains_float(self, value: float) -> bool:
        return value in self._float_atts

    def get_float_value(self, value: float) -> FloatAttribute | None:
        att = self._float_atts.get(value)
        if att is None and self._desc.min <= value <= self._desc.max:
            att = FloatAttribute(self._desc, value)
            self._float_atts[value] = att
        return att

    def get_floats(self):
        return self._float_atts.values()

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        assert self.project is not None
        if self.project.is_special_attribute(str(obj)):
            return self.project.get_special_attribute(str(obj))
        if isinstance(obj, (int, float)):
            return self.get_float_value(float(obj))
        return self.parse_value(str(obj))

    def get_value(self, value: float) -> SimpleAttribute | None:
        return self.get_float_value(value)

    def clean_data(self) -> None:
        for v in [
            v for v in self._float_atts
            if v < self._desc.min or v > self._desc.max
        ]:
            del self._float_atts[v]

    @override
    def parse_value(self, string: str):
        return self.get_float_value(float(string))
