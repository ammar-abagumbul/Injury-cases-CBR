from __future__ import annotations

from typing import TYPE_CHECKING

from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.similarity.taxonomy_node import TaxonomyNode


class Taxonomy:
    """A tree of symbols (used by TaxonomyFct). Each node has a similarity
    that says how similar its direct children are; leaves default to 1.00,
    the root defaults to 0.00."""

    def __init__(self, top_symbol: TaxonomyNode):
        self.top_symbol = top_symbol
        self.top_similarity = Similarity.get(0.00)
        self.parents: dict[TaxonomyNode, TaxonomyNode] = {}
        self.sims: dict[TaxonomyNode, Similarity] = {top_symbol: self.top_similarity}
        for att in top_symbol.nodes:
            self.sims[att] = Similarity.get(1.00)
            self.parents[att] = top_symbol
        self.leaves: list[TaxonomyNode] = list(top_symbol.nodes)

    def contains_as_subnode(self, top: TaxonomyNode | None, node: TaxonomyNode | None) -> bool:
        if top is None:
            return False
        if top is node:
            return True
        if top is self.top_symbol:
            return node in self.top_symbol.nodes
        while node is not top and node is not self.top_symbol:
            node = self.parents.get(node)  # type: ignore[assignment]
            if node is None:
                return False
        return node is top

    def set_parent(self, symbol: TaxonomyNode, new_parent: TaxonomyNode) -> None:
        if symbol is None or new_parent is None or self.contains_as_subnode(symbol, new_parent):
            return
        old_parent = self.parents.get(symbol)
        self.parents[symbol] = new_parent
        if old_parent is not None and old_parent not in self.parents.values():
            self.leaves.append(old_parent)
            self.sims[old_parent] = Similarity.get(1.00)
        if new_parent in self.leaves:
            self.leaves.remove(new_parent)

    def get_common_ancestor(self, symbol_att1: TaxonomyNode, symbol_att2: TaxonomyNode) -> TaxonomyNode:
        if symbol_att1 == symbol_att2:
            return symbol_att1
        parents1: list[TaxonomyNode] = []
        parents2: list[TaxonomyNode] = []
        self._get_parents_of(symbol_att1, parents1)
        self._get_parents_of(symbol_att2, parents2)

        i, j = len(parents1) - 1, len(parents2) - 1
        while parents1[i] == parents2[j]:
            i -= 1
            j -= 1
            if i == -1 or j == -1:
                break
        return parents1[i + 1]

    def _get_parents_of(self, node: TaxonomyNode, parents: list[TaxonomyNode]) -> None:
        parents.append(node)
        current_parent = self.get_parent(node)
        if current_parent is not None:
            self._get_parents_of(current_parent, parents)

    def is_leaf(self, att: TaxonomyNode) -> bool:
        return att in self.leaves

    def get_similarity(self, obj: TaxonomyNode) -> Similarity:
        return self.sims.get(obj, Similarity.INVALID_SIM)

    def get_leaves(self, top: TaxonomyNode | None = None) -> list[TaxonomyNode]:
        if top is None or top is self.top_symbol:
            return self.leaves
        return [leaf for leaf in self.leaves if self.contains_as_subnode(top, leaf)]

    def count_leaves(self, top: TaxonomyNode) -> int:
        if top is self.top_symbol:
            return len(self.leaves)
        return sum(1 for leaf in self.leaves if self.contains_as_subnode(top, leaf))

    def get_top_symbol(self) -> TaxonomyNode:
        return self.top_symbol

    def set_similarity(self, att: TaxonomyNode, sim: Similarity) -> None:
        if att is None or sim is None:
            return
        if att is self.top_symbol:
            self.sims[att] = sim
            self.top_similarity = sim
        elif att in self.top_symbol.nodes:
            self.sims[att] = sim

    def get_parent(self, current_node: TaxonomyNode) -> TaxonomyNode | None:
        return self.parents.get(current_node)

    def get_children(self, node: TaxonomyNode) -> list[TaxonomyNode]:
        return [child for child, parent in self.parents.items() if parent == node]

    def remove_attribute(self, a: TaxonomyNode) -> None:
        parent = self.get_parent(a)
        for child in self.get_children(a):
            if parent is not None:
                self.set_parent(child, parent)
        self.parents.pop(a, None)
        self.sims.pop(a, None)

    def add_attribute(self, a: TaxonomyNode) -> None:
        self.sims[a] = Similarity.get(1.00)
        self.leaves.append(a)
