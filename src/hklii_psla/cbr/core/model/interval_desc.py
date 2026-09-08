from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.interval_attribute import IntervalAttribute
    from hklii_psla.cbr.core.casebase.interval_range import IntervalRange
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.interval_fct import IntervalFct


class IntervalDesc(SimpleAttDesc):

    def __init__(self, owner: Concept, name: str, min_value: float, max_value: float):
        super().__init__(owner, name)
        if min_value > max_value:
            raise ValueError("min has to be less than or equal max")
        self.min = min_value
        self.max = max_value

        from hklii_psla.cbr.core.casebase.interval_range import IntervalRange

        self.range: IntervalRange = IntervalRange(owner.project, self)
        if owner is not None and owner is not owner.project:
            owner.add_attribute_desc(self)
        self.add_default_fct()

    @property
    def interval_range(self) -> IntervalRange:
        return self.range  # type: ignore[return-value]

    def get_interval_attribute(self, min_value: float, max_value: float) -> IntervalAttribute | None:
        return self.interval_range.get_interval_value((min_value, max_value))

    def set_min(self, min_value: float) -> None:
        if self.min != min_value and min_value < self.max:
            self.min = min_value
            self.owner.project.clean_instances(self.owner, self)

    def set_max(self, max_value: float) -> None:
        if self.max != max_value and max_value > self.min:
            self.max = max_value
            self.owner.project.clean_instances(self.owner, self)

    def add_interval_fct(self, name: str, active: bool) -> IntervalFct:
        from hklii_psla.cbr.core.similarity.interval_fct import IntervalFct

        f = IntervalFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        if isinstance(desc, IntervalDesc):
            return desc.min <= self.min and desc.max >= self.max
        return False

    def fits(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.interval_attribute import IntervalAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits(att):
            return False
        if isinstance(att, IntervalAttribute):
            return self._check(att)
        if isinstance(att, MultipleAttribute):
            for a in att.values:
                if not isinstance(a, IntervalAttribute) or not self._check(a):
                    return False
        return True

    def fits_single(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.interval_attribute import IntervalAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if not super().fits_single(att):
            return False
        if self.is_multiple and not isinstance(att, SpecialAttribute):
            if isinstance(att, IntervalAttribute):
                return self._check(att)
            return False
        return self.fits(att)

    def _check(self, i: IntervalAttribute) -> bool:
        lo, hi = i.interval
        return not (lo < self.min or hi > self.max)

    @override
    def add_default_fct(self) -> None:
        if self.owner is not None and self.owner is not self.owner.project:
            active_sim = self.add_interval_fct(self.owner.project.DEFAULT_FCT_NAME, False)
            self.update_amalgamation_fcts(self.owner, active_sim)
