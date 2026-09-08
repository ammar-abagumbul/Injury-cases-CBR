from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig
from hklii_psla.cbr.core.similarity.sim_fct import SimFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.date_desc import DateDesc
    from hklii_psla.cbr.core.project import Project


class DateFunctionPrecision(str, Enum):
    SECOND = "Second"
    MINUTE = "Minute"
    HOUR = "Hour"
    DAY = "Day"
    MONTH = "Month"
    YEAR = "Year"


class DateFct(SimFct):

    def __init__(self, prj: Project, desc: DateDesc, name: str, precision: DateFunctionPrecision):
        super().__init__(prj, desc, name)
        self.sub_desc = desc
        self.mc = MultipleConfig.DEFAULT_CONFIG
        self.precision = precision

    @override
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.date_attribute import DateAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if isinstance(a1, SpecialAttribute) or isinstance(a2, SpecialAttribute):
            return self.prj.calculate_special_similarity(a1, a2)
        if isinstance(a1, MultipleAttribute) and isinstance(a2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, a1, a2)
        if not (isinstance(a1, DateAttribute) and isinstance(a2, DateAttribute)):
            return Similarity.INVALID_SIM

        d1, d2 = a1.date, a2.date
        result = 0.0
        p = self.precision
        if p == DateFunctionPrecision.SECOND:
            if (d1.year, d1.month, d1.day, d1.hour, d1.minute) == (d2.year, d2.month, d2.day, d2.hour, d2.minute):
                result = 1 - abs(d2.second - d1.second) / 60.0
        elif p == DateFunctionPrecision.MINUTE:
            if (d1.year, d1.month, d1.day, d1.hour) == (d2.year, d2.month, d2.day, d2.hour):
                result = 1 - abs(d2.minute - d1.minute) / 60.0
        elif p == DateFunctionPrecision.HOUR:
            if (d1.year, d1.month, d1.day) == (d2.year, d2.month, d2.day):
                result = 1 - abs(d2.hour - d1.hour) / 24.0
        elif p == DateFunctionPrecision.DAY:
            if (d1.year, d1.month) == (d2.year, d2.month):
                result = 1 - abs(d2.day - d1.day) / 31.0
        elif p == DateFunctionPrecision.MONTH:
            if d1.year == d2.year:
                result = 1 - abs(d2.month - d1.month) / 12.0
        elif p == DateFunctionPrecision.YEAR:
            result = 1 - abs(d2.year - d1.year) / 100.0

        return Similarity.get(result)

    def get_multiple_config(self) -> MultipleConfig:
        return self.mc

    def set_multiple_config(self, mc: MultipleConfig) -> None:
        self.mc = mc

    def get_precision(self) -> DateFunctionPrecision:
        return self.precision

    def set_precision(self, precision: DateFunctionPrecision) -> None:
        self.precision = precision

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.date_desc import DateDesc
        from hklii_psla.cbr.core.project import Project

        if isinstance(new_desc, DateDesc) and self.name != Project.DEFAULT_FCT_NAME:
            f = new_desc.add_date_fct(self.name, is_active, self.precision)
            f.mc = self.mc
