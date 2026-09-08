from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.string_attribute import StringAttribute
    from hklii_psla.cbr.core.casebase.string_range import StringRange
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.config.string_config import StringConfig
    from hklii_psla.cbr.core.similarity.string_fct import StringFct


class StringDesc(SimpleAttDesc):

    def __init__(self, owner: Concept, name: str):
        super().__init__(owner, name)

        from hklii_psla.cbr.core.casebase.string_range import StringRange

        self.range: StringRange = StringRange(owner.project, self)
        if owner is not None and owner is not owner.project:
            owner.add_attribute_desc(self)
        self.add_default_fct()

    @property
    def string_range(self) -> StringRange:
        return self.range  # type: ignore[return-value]

    def get_string_attribute(self, value: str) -> StringAttribute:
        return self.string_range.get_string_value(value)

    def add_string_fct(self, config: StringConfig, name: str, active: bool) -> StringFct:
        from hklii_psla.cbr.core.similarity.string_fct import StringFct

        f = StringFct(self.owner.project, config, self, name)
        self.add_function(f, active)
        return f

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        return isinstance(desc, StringDesc)

    @override
    def add_default_fct(self) -> None:
        from hklii_psla.cbr.core.similarity.config.string_config import StringConfig

        active_sim = self.add_string_fct(
            StringConfig.EQUALITY, self.owner.project.DEFAULT_FCT_NAME, False
        )
        self.update_amalgamation_fcts(self.owner, active_sim)
