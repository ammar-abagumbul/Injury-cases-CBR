from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.date_attribute import DateAttribute
    from hklii_psla.cbr.core.casebase.date_range import DateRange
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.date_fct import DateFct, DateFunctionPrecision


class DateDesc(SimpleAttDesc):

    def __init__(
        self,
        owner: Concept,
        name: str,
        min_date: datetime,
        max_date: datetime,
        date_format: str = "%Y-%m-%d",
    ):
        super().__init__(owner, name)
        if min_date > max_date:
            raise ValueError("min has to be before max date")
        self.format = date_format
        self.min_date = min_date
        self.max_date = max_date

        from hklii_psla.cbr.core.casebase.date_range import DateRange

        self.range: DateRange = DateRange(owner.project, self)
        if owner is not None and owner is not owner.project:
            owner.add_attribute_desc(self)
        self.add_default_fct()

    @property
    def date_range(self) -> DateRange:
        return self.range  # type: ignore[return-value]

    def get_date_attribute(self, value: datetime) -> DateAttribute:
        return self.date_range.get_date_value(value)

    def set_min_date(self, min_date: datetime) -> None:
        if self.min_date != min_date and min_date < self.max_date:
            self.min_date = min_date
            self.owner.project.clean_instances(self.owner, self)

    def set_max_date(self, max_date: datetime) -> None:
        if self.max_date != max_date and max_date > self.min_date:
            self.max_date = max_date
            self.owner.project.clean_instances(self.owner, self)

    def add_date_fct(
        self,
        name: str,
        active: bool,
        precision: DateFunctionPrecision | None = None,
    ) -> DateFct:
        from hklii_psla.cbr.core.similarity.date_fct import DateFct, DateFunctionPrecision

        f = DateFct(self.owner.project, self, name, precision or DateFunctionPrecision.YEAR)
        self.add_function(f, active)
        return f

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        if isinstance(desc, DateDesc):
            return desc.min_date <= self.min_date and desc.max_date >= self.max_date
        return False

    def fits(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.date_attribute import DateAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits(att):
            return False
        if isinstance(att, DateAttribute):
            return self._check(att)
        if isinstance(att, MultipleAttribute):
            for a in att.values:
                if not isinstance(a, DateAttribute) or not self._check(a):
                    return False
        return True

    def fits_single(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.date_attribute import DateAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits_single(att):
            return False
        if self.is_multiple and not isinstance(att, MultipleAttribute):
            if isinstance(att, DateAttribute):
                return self._check(att)
            return False
        return self.fits(att)

    def _check(self, a: DateAttribute) -> bool:
        return not (a.date < self.min_date or a.date > self.max_date)

    @override
    def add_default_fct(self) -> None:
        from hklii_psla.cbr.core.similarity.date_fct import DateFunctionPrecision

        if self.owner is not None and self.owner is not self.owner.project:
            active_sim = self.add_date_fct(
                self.owner.project.DEFAULT_FCT_NAME, False, DateFunctionPrecision.DAY
            )
            self.update_amalgamation_fcts(self.owner, active_sim)
