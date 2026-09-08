from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
from hklii_psla.cbr.core.casebase.symbol_range import SymbolRange

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.special_desc import SpecialDesc
    from hklii_psla.cbr.core.project import Project


class SpecialRange(SymbolRange):
    """Range holding all possible special values of the project."""

    def __init__(self, prj: Project, desc: SpecialDesc, allowed_values: Collection[str]):
        super().__init__(prj, desc, allowed_values)
        self.init_symbol_attributes(allowed_values)
        self.init_indexes()

    @override
    def init_symbol_attributes(self, values: Collection[str]) -> None:
        self.symbols = {
            value: SpecialAttribute(self.project, self.desc, value)  # type: ignore[arg-type]
            for value in values
        }

    def add_symbol_value(self, value: str) -> SpecialAttribute:
        att = self.symbols.get(value)
        if att is None:
            att = SpecialAttribute(self.project, self.desc, value)  # type: ignore[arg-type]
            self.symbols[value] = att
            self._indexes[att] = self.highest_index
            self.highest_index += 1
        return att  # type: ignore[return-value]

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        assert isinstance(obj, str)
        if self.project is not None and self.project.is_special_attribute(obj):
            return self.get_symbol_value(obj)
        return None
