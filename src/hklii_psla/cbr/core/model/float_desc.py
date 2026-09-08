from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.float_attribute import FloatAttribute
    from hklii_psla.cbr.core.casebase.float_range import FloatRange
    from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.advanced_float_fct import AdvancedFloatFct
    from hklii_psla.cbr.core.similarity.float_fct import FloatFct


class FloatDesc(SimpleAttDesc):

    def __init__(self, owner: Concept, name: str, min_value: float, max_value: float):
        super().__init__(owner, name)
        if min_value > max_value:
            raise ValueError("min has to be less than or equal max")
        self.min = min_value
        self.max = max_value

        from hklii_psla.cbr.core.casebase.float_range import FloatRange

        self.range: FloatRange = FloatRange(owner.project, self)
        if owner is not None and owner is not owner.project:
            owner.add_attribute_desc(self)
        self.add_default_fct()

    @property
    def float_range(self) -> FloatRange:
        return self.range  # type: ignore[return-value]

    def get_number_attribute(self, value: float) -> SimpleAttribute | None:
        return self.float_range.get_value(value)

    def set_min(self, min_value: float) -> None:
        if self.min != min_value and min_value < self.max:
            self.min = min_value
            self.owner.project.clean_instances(self.owner, self)

    def set_max(self, max_value: float) -> None:
        if self.max != max_value and max_value > self.min:
            self.max = max_value
            self.owner.project.clean_instances(self.owner, self)

    def add_float_fct(self, name: str, active: bool) -> FloatFct:
        from hklii_psla.cbr.core.similarity.float_fct import FloatFct

        f = FloatFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    def add_advanced_float_fct(self, name: str, active: bool) -> AdvancedFloatFct:
        from hklii_psla.cbr.core.similarity.advanced_float_fct import AdvancedFloatFct

        f = AdvancedFloatFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        if isinstance(desc, FloatDesc):
            return desc.min <= self.min and desc.max >= self.max
        return False

    def fits(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.float_attribute import FloatAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits(att):
            return False
        if isinstance(att, FloatAttribute):
            return self._check(att)
        if isinstance(att, MultipleAttribute):
            for a in att.values:
                if not isinstance(a, FloatAttribute) or not self._check(a):
                    return False
        return True

    def fits_single(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.float_attribute import FloatAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits_single(att):
            return False
        if self.is_multiple and not isinstance(att, MultipleAttribute):
            if isinstance(att, FloatAttribute):
                return self._check(att)
            return False
        return self.fits(att)

    def _check(self, i: FloatAttribute) -> bool:
        return not (i.value < self.min or i.value > self.max)

    @override
    def add_default_fct(self) -> None:
        active_sim = self.add_float_fct(self.owner.project.DEFAULT_FCT_NAME, False)
        self.update_amalgamation_fcts(self.owner, active_sim)
