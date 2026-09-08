from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
    from hklii_psla.cbr.core.project import Project
    from hklii_psla.cbr.core.similarity.integer_fct import IntegerFct
    from hklii_psla.cbr.core.similarity.similarity import Similarity


class OrderedSymbolFct(SymbolFct):
    """Provides a linear (optionally cyclic) order over the known symbol
    attributes, and compares them by the distance between their positions."""

    def __init__(self, prj: Project, desc: SymbolDesc, name: str):
        super().__init__(prj, desc, name)
        from hklii_psla.cbr.core.model.integer_desc import IntegerDesc
        from hklii_psla.cbr.core.similarity.integer_fct import IntegerFct

        self.order: dict[SymbolAttribute, int] = {}
        self.is_cyclic = False
        self.highest_order = 1
        self.min_order = 1
        self.distance_last_first = 1

        self.internal_desc = IntegerDesc(prj, f"{name}Internal", 0, len(desc.get_allowed_values()))
        self.internal_function = IntegerFct(prj, self.internal_desc, f"{name}Internal")

        for i, att in enumerate(desc.get_symbol_attributes(), start=1):
            self.order[att] = i
            self.highest_order = i
        self.internal_desc.min = self.min_order
        self.internal_desc.max = self.highest_order

    def update_table(self) -> None:
        for att1 in self.sub_desc.get_symbol_attributes():
            for att2 in self.sub_desc.get_symbol_attributes():
                query_int = self.order.get(att1)
                case_int = self.order.get(att2)
                if query_int is None or case_int is None:
                    continue

                if self.is_cyclic:
                    rng = self.min_order + self.distance_last_first - self.highest_order
                    diff1 = query_int - case_int
                    if diff1 < -(rng // 2):
                        query_int = query_int + rng
                    elif diff1 > (rng // 2):
                        query_int = query_int - rng

                sim = self.internal_function.calculate_similarity_values(query_int, case_int)
                super().set_similarity(att1, att2, sim)

    def _change_range(self, minimum: int, maximum: int, is_cyclic: bool, dist_last_first: int) -> None:
        self.min_order = minimum
        self.highest_order = maximum
        self.is_cyclic = is_cyclic
        self.distance_last_first = dist_last_first

        if is_cyclic:
            self.internal_desc.max = self.min_order + self.highest_order
            self.internal_desc.min = self.min_order
        else:
            self.internal_desc.max = self.highest_order
            self.internal_desc.min = self.min_order

    def get_order(self) -> dict[SymbolAttribute, int]:
        return self.order

    def get_internal_function(self) -> IntegerFct:
        return self.internal_function

    def remove_symbol(self, att: SymbolAttribute) -> None:
        self.order.pop(att, None)

    def add_symbol(self, att: SymbolAttribute) -> None:
        if self.sub_desc.is_allowed_value(att.value) and att not in self.order:
            self.highest_order += 1
            self.order[att] = self.highest_order
            super().add_attribute(att)
            self.internal_desc.max = self.highest_order

    def set_order_index_of(self, att: SymbolAttribute | str, index: int) -> None:
        if isinstance(att, str):
            resolved = self.desc.get_attribute(att)
            from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute as SA

            if not isinstance(resolved, SA):
                return
            att = resolved

        if not self.sub_desc.is_allowed_value(att.value):
            return

        new_highest = max([index, *self.order.values()])
        new_min = min([index, *self.order.values()])
        update_highest = new_highest != self.highest_order
        update_min = new_min != self.min_order
        self.highest_order = new_highest
        self.min_order = new_min

        self.order[att] = index
        if update_min or update_highest:
            self._change_range(self.min_order, self.highest_order, self.is_cyclic, self.distance_last_first)
        self.update_table()

    def is_cyclic_(self) -> bool:
        return self.is_cyclic

    def set_cyclic(self, cyclic: bool) -> None:
        self.is_cyclic = cyclic
        self._change_range(self.min_order, self.highest_order, cyclic, self.distance_last_first)

    # these overrides no-op because the table is derived from `order`, not set directly.
    def set_similarity(self, *args, **kwargs) -> bool:  # type: ignore[override]
        return False

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
        from hklii_psla.cbr.core.project import Project

        if type(new_desc) is type(self.desc) and self.name != Project.DEFAULT_FCT_NAME:
            assert isinstance(new_desc, SymbolDesc)
            f = new_desc.add_ordered_symbol_fct(self.name, is_active)
            f.sims = [row[:] for row in self.sims]
            f.is_symmetric = self.is_symmetric
            f.mc = self.mc
            f.highest_order = self.highest_order
            f.internal_desc = self.internal_desc
            f.internal_function = self.internal_function
            f.is_cyclic = self.is_cyclic
            f.min_order = self.min_order
            f.order = self.order
            f.distance_last_first = self.distance_last_first

    def set_distance_last_first(self, distance_last_first: int) -> None:
        self.distance_last_first = distance_last_first
        self._change_range(self.min_order, self.highest_order, self.is_cyclic, distance_last_first)

    def get_distance_last_first(self) -> int:
        return self.distance_last_first
