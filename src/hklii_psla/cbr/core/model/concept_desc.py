from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.amalgamation_fct import AmalgamationFct
    from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig


class ConceptDesc(AttributeDesc):
    """An attribute description whose values are instances of another concept (composition)."""

    def __init__(self, owner: Concept, name: str, concept: Concept):
        super().__init__(owner, name)
        from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig

        self.concept = concept
        self.concept.add_part_of_relation(self)
        if owner is not None and owner is not owner.project:
            owner.add_attribute_desc(self)
        self.range = concept.range
        active_sim = concept.get_active_amalgam_fct()
        self.update_amalgamation_fcts(owner, active_sim)

        self._multiple_configs: dict[AmalgamationFct, MultipleConfig] = {
            f: MultipleConfig.DEFAULT_CONFIG for f in concept.get_available_amalgam_fcts()
        }

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        if isinstance(desc, ConceptDesc):
            return self.concept.can_override(desc.concept)
        return False

    def set_concept(self, c: Concept) -> None:
        self.concept.delete_part_of_relation(self)
        self.concept = c
        c.add_part_of_relation(self)
        self.range = c.range
        self.owner.project.clean_instances(self.owner, self)

    def fits(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.instance import Instance
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        if not super().fits(att):
            return False
        if isinstance(att, Instance):
            return self._check(att)
        if isinstance(att, MultipleAttribute):
            for a in att.values:
                if not isinstance(a, Instance) or not self._check(a):
                    return False
        return True

    def fits_single(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.instance import Instance
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if not super().fits_single(att):
            return False
        if self.is_multiple and not isinstance(att, SpecialAttribute):
            if isinstance(att, Instance):
                return self._check(att)
            return False
        return self.fits(att)

    def _check(self, i: Instance) -> bool:
        for desc, value in i.attributes.items():
            if not desc.fits(value):
                return False
        return True

    @override
    def delete_all_fcts(self) -> None:
        if self.concept is None:
            return
        self.concept.get_available_amalgam_fcts().clear()

    def get_multiple_config(self, f: AmalgamationFct) -> MultipleConfig:
        from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig

        return self._multiple_configs.get(f, MultipleConfig.DEFAULT_CONFIG)

    def set_multiple_config(self, f: AmalgamationFct, mc: MultipleConfig) -> None:
        self._multiple_configs[f] = mc
