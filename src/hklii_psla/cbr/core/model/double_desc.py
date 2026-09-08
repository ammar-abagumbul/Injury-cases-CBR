from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.double_attribute import DoubleAttribute
    from hklii_psla.cbr.core.casebase.double_range import DoubleRange
    from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.advanced_double_fct import AdvancedDoubleFct
    from hklii_psla.cbr.core.similarity.double_fct import DoubleFct


class DoubleDesc(SimpleAttDesc):

    def __init__(self, owner: Concept, name: str, min_value: float, max_value: float):
        super().__init__(owner, name)
        if min_value > max_value:
            raise ValueError("min has to be less than or equal max")
        self.min = min_value
        self.max = max_value

        from hklii_psla.cbr.core.casebase.double_range import DoubleRange

        self.range: DoubleRange = DoubleRange(owner.project, self)
        if owner is not None and owner is not owner.project:
            owner.add_attribute_desc(self)
        self.add_default_fct()

    @property
    def double_range(self) -> DoubleRange:
        return self.range  # type: ignore[return-value]

    def get_number_attribute(self, value: float) -> SimpleAttribute | None:
        return self.double_range.get_value(value)

    def set_min(self, min_value: float) -> None:
        if self.min != min_value and min_value < self.max:
            self.min = min_value
            self.owner.project.clean_instances(self.owner, self)

    def set_max(self, max_value: float) -> None:
        if self.max != max_value and max_value > self.min:
            self.max = max_value
            self.owner.project.clean_instances(self.owner, self)

    def add_double_fct(self, name: str, active: bool) -> DoubleFct:
        from hklii_psla.cbr.core.similarity.double_fct import DoubleFct

        f = DoubleFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    def add_advanced_double_fct(self, name: str, active: bool) -> AdvancedDoubleFct:
        from hklii_psla.cbr.core.similarity.advanced_double_fct import AdvancedDoubleFct

        f = AdvancedDoubleFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        if isinstance(desc, DoubleDesc):
            return desc.min <= self.min and desc.max >= self.max
        return False

    def fits(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.double_attribute import DoubleAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits(att):
            return False
        if isinstance(att, DoubleAttribute):
            return self._check(att)
        if isinstance(att, MultipleAttribute):
            for a in att.values:
                if not isinstance(a, DoubleAttribute) or not self._check(a):
                    return False
        return True

    def fits_single(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.double_attribute import DoubleAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits_single(att):
            return False
        if self.is_multiple and not isinstance(att, MultipleAttribute):
            if isinstance(att, DoubleAttribute):
                return self._check(att)
            return False
        return self.fits(att)

    def _check(self, i: DoubleAttribute) -> bool:
        return not (i.value < self.min or i.value > self.max)

    @override
    def add_default_fct(self) -> None:
        active_sim = self.add_double_fct(self.owner.project.DEFAULT_FCT_NAME, False)
        self.update_amalgamation_fcts(self.owner, active_sim)
