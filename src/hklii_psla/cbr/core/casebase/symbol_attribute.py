from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
from hklii_psla.cbr.core.similarity.taxonomy_node import TaxonomyNode

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc


class SymbolAttribute(SimpleAttribute, TaxonomyNode):

    def __init__(
        self,
        desc: SymbolDesc,
        v: str
    ):
        super().__init__(desc)
        self._value = v
        self._desc: SymbolDesc = desc

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, v: str) -> None:
        if self._desc.is_allowed_value(v):
            return
        if self._desc.rename_value(self._value, v):
            self._value = v
        self._value = v

    @override
    def get_value_as_string(self) -> str:
        return self._value

    @property
    @override
    def nodes(self) -> list[TaxonomyNode]:
        return list(self._desc.get_symbol_attributes())

    def __repr__(self) -> str:
        return self._value
