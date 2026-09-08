from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig
from hklii_psla.cbr.core.similarity.sim_fct import SimFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.interval_desc import IntervalDesc
    from hklii_psla.cbr.core.project import Project


class IntervalFct(SimFct):
    """Intervals are similar (1.0) iff their bounds are equal, else 0.0."""

    def __init__(self, prj: Project, desc: IntervalDesc, name: str):
        super().__init__(prj, desc, name)
        self.sub_desc = desc
        self.mc = MultipleConfig.DEFAULT_CONFIG

    @override
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.interval_attribute import IntervalAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if isinstance(a1, SpecialAttribute) or isinstance(a2, SpecialAttribute):
            return self.prj.calculate_special_similarity(a1, a2)
        if isinstance(a1, MultipleAttribute) and isinstance(a2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, a1, a2)
        if isinstance(a1, IntervalAttribute) and isinstance(a2, IntervalAttribute):
            return Similarity.get(1.0) if a1 == a2 else Similarity.get(0.0)
        return Similarity.INVALID_SIM

    def get_multiple_config(self) -> MultipleConfig:
        return self.mc

    def set_multiple_config(self, mc: MultipleConfig) -> None:
        self.mc = mc

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.interval_desc import IntervalDesc
        from hklii_psla.cbr.core.project import Project

        if isinstance(new_desc, IntervalDesc) and self.name != Project.DEFAULT_FCT_NAME:
            f = new_desc.add_interval_fct(self.name, is_active)
            f.mc = self.mc
