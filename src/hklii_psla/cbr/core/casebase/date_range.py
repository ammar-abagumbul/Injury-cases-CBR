from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.date_attribute import DateAttribute
from hklii_psla.cbr.core.casebase.range import Range

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.date_desc import DateDesc
    from hklii_psla.cbr.core.project import Project


class DateRange(Range):

    def __init__(self, p: Project, d: DateDesc):
        super().__init__(p)
        self._desc = d
        self._atts: dict[datetime, DateAttribute] = {}

    def get_date_value(self, date: datetime) -> DateAttribute | None:
        att = self._atts.get(date)
        if att is None and self._desc.min_date <= date <= self._desc.max_date:
            att = DateAttribute(self._desc, date)
            self._atts[date] = att
        return att

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        assert self.project is not None
        if isinstance(obj, str) and self.project.is_special_attribute(obj):
            return self.project.get_special_attribute(obj)
        if isinstance(obj, datetime):
            return self.get_date_value(obj)
        return self.parse_value(str(obj))

    @override
    def parse_value(self, string: str):
        return self.get_date_value(datetime.strptime(string, self._desc.format))

    def clean_data(self) -> None:
        for d in [
            d for d, att in self._atts.items()
            if att.date < self._desc.min_date or att.date > self._desc.max_date
        ]:
            del self._atts[d]
