from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
    from hklii_psla.cbr.core.casebase.special_range import SpecialRange
    from hklii_psla.cbr.core.project import Project
    from hklii_psla.cbr.core.similarity.special_fct import SpecialFct


class SpecialDesc(SymbolDesc):
    """Special values ("undefined"/"unknown"/"others") usable for any attribute description."""

    def __init__(self, project: Project, allowed_values: Collection[str]):
        from hklii_psla.cbr.core.casebase.special_range import SpecialRange

        super().__init__(project, "specialValueDesc", set(allowed_values))
        self.range: SpecialRange = SpecialRange(project, self, allowed_values)

    @property
    def special_range(self) -> SpecialRange:
        return self.range  # type: ignore[return-value]

    def get_attribute(self, value: object) -> Attribute | None:
        assert isinstance(value, str)
        if self.owner.project.is_special_attribute(value):
            return self.special_range.get_attribute(value)
        return None

    def get_index_of(self, att: SpecialAttribute) -> int | None:
        return self.special_range.get_index_of(att)

    def remove_symbol(self, value: str) -> None:
        from hklii_psla.cbr.core.project import Project

        if value != Project.UNDEFINED_SPECIAL_ATTRIBUTE:
            super().remove_symbol(value)

    def add_special_fct(self, name: str, active: bool) -> SpecialFct:
        from hklii_psla.cbr.core.similarity.special_fct import SpecialFct

        f = SpecialFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f
