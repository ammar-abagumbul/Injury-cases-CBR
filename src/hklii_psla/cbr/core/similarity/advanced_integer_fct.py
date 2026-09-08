from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.advanced_double_fct import AdvancedDoubleFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.integer_desc import IntegerDesc
    from hklii_psla.cbr.core.project import Project


class AdvancedIntegerFct(AdvancedDoubleFct):
    """Same piecewise-linear interpolation as AdvancedDoubleFct, for IntegerDesc."""

    def __init__(self, prj: Project, desc: IntegerDesc, name: str):
        super().__init__(prj, desc, name)  # type: ignore[arg-type]

    @override
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.integer_attribute import IntegerAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if isinstance(a1, SpecialAttribute) or isinstance(a2, SpecialAttribute):
            return self.prj.calculate_special_similarity(a1, a2)
        if isinstance(a1, MultipleAttribute) and isinstance(a2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, a1, a2)
        if isinstance(a1, SimpleAttribute) and isinstance(a2, SimpleAttribute):
            if a1.attribute_desc is not self.desc or a2.attribute_desc is not self.desc:
                return Similarity.INVALID_SIM
            if isinstance(a1, IntegerAttribute) and isinstance(a2, IntegerAttribute):
                return self.calculate_similarity_values(a1.value, a2.value)
        return Similarity.INVALID_SIM

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.integer_desc import IntegerDesc
        from hklii_psla.cbr.core.project import Project

        if isinstance(new_desc, IntegerDesc) and self.name != Project.DEFAULT_FCT_NAME:
            f = new_desc.add_advanced_integer_fct(self.name, is_active)
            f.distance_function = self.distance_function
            f._points = dict(self._points)
            f.zero_point = self.zero_point
            f.is_symmetric = self.is_symmetric
