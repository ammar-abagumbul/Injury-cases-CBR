from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.date_desc import DateDesc


class DateAttribute(SimpleAttribute):

    def __init__(self, desc: DateDesc, d: datetime):
        super().__init__(desc)
        self.date = d

    @override
    def get_value_as_string(self) -> str:
        return self.date.isoformat()
