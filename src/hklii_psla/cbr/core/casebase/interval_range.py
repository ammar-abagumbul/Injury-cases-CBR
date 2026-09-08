from __future__ import annotations

import re
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.interval_attribute import IntervalAttribute
from hklii_psla.cbr.core.casebase.range import Range

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.interval_desc import IntervalDesc
    from hklii_psla.cbr.core.project import Project


class IntervalRange(Range):

    def __init__(self, prj: Project, d: IntervalDesc):
        super().__init__(prj)
        self._desc = d
        self._atts: dict[tuple[float, float], IntervalAttribute] = {}

    def get_interval_value(self, interval: tuple[float, float]) -> IntervalAttribute | None:
        att = self._atts.get(interval)
        if att is None and interval[0] >= self._desc.min and interval[1] <= self._desc.max:
            att = IntervalAttribute(self._desc, interval)
            self._atts[interval] = att
        return att

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        assert self.project is not None
        if isinstance(obj, str) and self.project.is_special_attribute(obj):
            return self.project.get_special_attribute(obj)
        if isinstance(obj, tuple):
            return self.get_interval_value(obj)
        return self.parse_value(str(obj))

    def clean_data(self) -> None:
        for interval in [
            i for i in self._atts
            if i[0] < self._desc.min or i[1] > self._desc.max
        ]:
            del self._atts[interval]

    @override
    def parse_value(self, string: str):
        match = re.match(r"\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", string)
        if match:
            return self.get_interval_value((float(match.group(1)), float(match.group(2))))
        return None
