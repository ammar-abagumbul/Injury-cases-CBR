from collections.abc import Collection, Set
from typing import override

from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
from hklii_psla.cbr.core.casebase.symbol_range import SymbolRange
from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.concept import Concept
from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc
from hklii_psla.cbr.core.similarity.taxonomy_node import TaxonomyNode


class SymbolDesc(SimpleAttDesc, TaxonomyNode):

    def __init__(
        self,
        owner: Concept,
        name: str,
        allowed_values: set[str]
    ):
        super().__init__(owner, name)
        self._range: SymbolRange

    def is_allowed_value(self, value: str) -> bool:
        return bool(
            self._range.symbols.get(value)
        )

    def rename_value(self, old_value: str, new_value: str):
        # WARN: renaming might fail, look into it
        if new_value.strip():
            self._range.rename_symbol(old_value, new_value)
            # TODO
            # set_changed()
            # notify_observers()
            return True
        return False

    @property
    @override
    def nodes(self) -> list[TaxonomyNode]:
        return []

    def get_symbol_attrs(self) -> Collection[SymbolAttribute]:
        return self._range.symbols.values()

    def get_allowed_values(self) -> set[str]:
        return set(self._range.symbols.keys())

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        return bool(
            desc
            and isinstance(desc, SymbolDesc)
            and desc.get_allowed_values()
            and desc.get_allowed_values().issuperset(self.get_allowed_values())
        )
