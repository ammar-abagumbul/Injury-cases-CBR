from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
    from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct


class BooleanDesc(SymbolDesc):
    """Reuses SymbolFct's table-based similarity for the two boolean values."""

    def __init__(self, owner: Concept, name: str):
        from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

        super().__init__(owner, name, set())
        self._symbol_true = SymbolAttribute(self, "true")
        self._symbol_false = SymbolAttribute(self, "false")

    def get_boolean_attribute(self, value: bool | None) -> SymbolAttribute | None:
        if value is True:
            return self._symbol_true
        if value is False:
            return self._symbol_false
        return None

    def add_boolean_fct(self, name: str, active: bool) -> SymbolFct:
        from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct

        f = SymbolFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        return isinstance(desc, BooleanDesc)

    def is_allowed_value(self, value: str) -> bool:
        return value in ("true", "false")

    @override
    def get_attribute(self, value: object) -> Attribute | None:
        if value is None:
            return None
        if self.owner.project.is_special_attribute(str(value)):
            return self.owner.project.get_special_attribute(str(value))
        if isinstance(value, str):
            return self.get_boolean_attribute(value.lower() == "true")
        return self.get_boolean_attribute(bool(value))

    def get_index_of(self, att: SimpleAttribute) -> int | None:
        if att is self._symbol_true:
            return 0
        if att is self._symbol_false:
            return 1
        return None

    def remove_symbol(self, value: str) -> None:
        pass

    def rename_value(self, old_value: str, new_value: str) -> bool:
        return False

    def add_symbol(self, value: str) -> SymbolAttribute | None:
        return None
