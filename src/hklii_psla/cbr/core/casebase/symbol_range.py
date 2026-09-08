from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.range import Range
from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
    from hklii_psla.cbr.core.project import Project


class SymbolRange(Range):

    symbols: dict[str, SymbolAttribute]

    def __init__(
        self,
        project: Project | None,
        description: SymbolDesc,
        allowed_values: Collection[str] | None,
    ):
        super().__init__(project)
        self._desc: SymbolDesc = description
        self._indexes: dict[SymbolAttribute, int] = {}
        self._highest_index: int = 0

        self.init_symbol_attributes(allowed_values or [])
        self.init_indexes()

    def init_symbol_attributes(self, values: Collection[str]) -> None:
        self.symbols = {}
        for value in values:
            if value.strip():
                self.symbols[value] = SymbolAttribute(self._desc, value)

    def init_indexes(self) -> None:
        self._indexes = {att: i for i, att in enumerate(self.symbols.values())}
        self._highest_index = len(self.symbols)

    def get_symbol_value(self, symbol: str) -> SymbolAttribute | None:
        return self.symbols.get(symbol)

    @override
    def get_attribute(self, obj: object):
        if isinstance(obj, str):
            if self.project is not None and self.project.is_special_attribute(obj):
                return self.project.get_special_attribute(obj)
            return self.get_symbol_value(obj)
        return None

    def get_index_of(self, att: SymbolAttribute | None) -> int | None:
        if att is None:
            return None
        return self._indexes.get(att)

    def remove_attribute(self, symbol: str) -> SymbolAttribute | None:
        att = self.symbols.pop(symbol, None)
        self.init_indexes()
        return att

    def rename_symbol(self, old_value: str, new_value: str) -> None:
        att = self.symbols.pop(old_value, None)
        if att is not None:
            self.symbols[new_value] = att

    def add_symbol_value(self, value: str) -> SymbolAttribute | None:
        if not value.strip():
            return None
        att = self.symbols.get(value)
        if att is None:
            att = SymbolAttribute(self._desc, value)
            self.symbols[value] = att
            self._indexes[att] = self._highest_index
            self._highest_index += 1
        return att

    @property
    def desc(self) -> SymbolDesc:
        return self._desc

    @property
    def indexes(self) -> dict[SymbolAttribute, int]:
        return self._indexes

    @property
    def highest_index(self) -> int:
        return self._highest_index

    @highest_index.setter
    def highest_index(self, value: int) -> None:
        self._highest_index = value

    @override
    def parse_value(self, string: str):
        return self.get_symbol_value(string)
