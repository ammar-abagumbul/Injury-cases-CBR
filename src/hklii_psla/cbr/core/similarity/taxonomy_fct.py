from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.taxonomy_config import TaxonomyConfig
from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct
from hklii_psla.cbr.core.similarity.taxonomy import Taxonomy

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
    from hklii_psla.cbr.core.project import Project
    from hklii_psla.cbr.core.similarity.similarity import Similarity
    from hklii_psla.cbr.core.similarity.taxonomy_node import TaxonomyNode


class TaxonomyFct(SymbolFct):
    """Similarity function for a symbol description structured as a tree,
    where each inner node carries the similarity of its children."""

    def __init__(self, prj: Project, top_symbol: SymbolDesc, values: list[SymbolAttribute], name: str):
        super().__init__(prj, top_symbol, name)
        self.query_config = TaxonomyConfig.INNER_NODES_ANY
        self.case_config = TaxonomyConfig.INNER_NODES_ANY
        self.taxonomy = Taxonomy(top_symbol)

    def update_table(self) -> None:
        nodes = self.taxonomy.get_top_symbol().nodes
        for att1 in nodes:
            for att2 in nodes:
                if self.query_config == TaxonomyConfig.NO_INNERNODES and self.case_config == TaxonomyConfig.NO_INNERNODES:
                    continue
                if self.query_config != TaxonomyConfig.NO_INNERNODES and self.case_config != TaxonomyConfig.NO_INNERNODES:
                    pass
                elif self.query_config != TaxonomyConfig.NO_INNERNODES:
                    if not self.taxonomy.is_leaf(att2):
                        continue
                elif self.case_config != TaxonomyConfig.NO_INNERNODES:
                    if not self.taxonomy.is_leaf(att1):
                        continue
                else:
                    if not (self.taxonomy.is_leaf(att1) and self.taxonomy.is_leaf(att2)):
                        continue
                sim = self._compute_similarity_by_taxonomy(att1, att2)
                self.set_similarity(str(att1), str(att2), sim)

    def _compute_similarity_by_taxonomy(self, att1: TaxonomyNode, att2: TaxonomyNode) -> Similarity:
        from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        if att1 == att2:
            return Similarity.get(1.00)
        if isinstance(att1, SymbolAttribute) and isinstance(att2, SymbolAttribute):
            if not (self.prj.is_special_attribute(att1.value) or self.prj.is_special_attribute(att2.value)):
                is_att1_leaf = self.taxonomy.is_leaf(att1)
                is_att2_leaf = self.taxonomy.is_leaf(att2)

                if is_att1_leaf and is_att2_leaf:
                    ancestor = self.taxonomy.get_common_ancestor(att1, att2)
                    return self.taxonomy.get_similarity(ancestor)
                if is_att1_leaf:
                    return self._similarity_for_inner_node(att2, att1, self.case_config)
                if is_att2_leaf:
                    return self._similarity_for_inner_node(att1, att2, self.query_config)

                if self.query_config == self.case_config:
                    if self.query_config in (
                        TaxonomyConfig.INNER_NODES_OPTIMISTIC,
                        TaxonomyConfig.INNER_NODES_ANY,
                    ):
                        if self.taxonomy.contains_as_subnode(
                            att1, att2
                        ) or self.taxonomy.contains_as_subnode(att2, att1):
                            return Similarity.get(1.00)
                        ancestor = self.taxonomy.get_common_ancestor(att1, att2)
                        return self.taxonomy.get_similarity(ancestor)
                    if self.query_config == TaxonomyConfig.INNER_NODES_PESSIMISTIC:
                        ancestor = self.taxonomy.get_common_ancestor(att1, att2)
                        return self.taxonomy.get_similarity(ancestor)
                    # INNER_NODES_AVERAGE
                    if not self.taxonomy.contains_as_subnode(att1, att2) and not self.taxonomy.contains_as_subnode(
                        att2, att1
                    ):
                        ancestor = self.taxonomy.get_common_ancestor(att1, att2)
                        return self.taxonomy.get_similarity(ancestor)
                    if att1 == att2:
                        child_count = self.taxonomy.count_leaves(att1)
                        sim = (1.0 + (child_count - 1) * self.taxonomy.get_similarity(att1).value) / child_count
                        return Similarity.get(sim)
                    factor = (1 / self.taxonomy.count_leaves(att1)) * (1 / self.taxonomy.count_leaves(att2))
                    result = 0.0
                    for child1 in self.taxonomy.get_leaves(att1):
                        for child2 in self.taxonomy.get_leaves(att2):
                            result += self.calculate_similarity(child1, child2).value  # type: ignore[arg-type]
                    return Similarity.get(result * factor)
            else:
                return self.prj.calculate_special_similarity(att1, att2)
        return Similarity.INVALID_SIM

    def _similarity_for_inner_node(
        self, att1: SymbolAttribute, att2: SymbolAttribute, config: TaxonomyConfig
    ) -> Similarity:
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        if config in (TaxonomyConfig.INNER_NODES_ANY, TaxonomyConfig.INNER_NODES_OPTIMISTIC):
            if self.taxonomy.contains_as_subnode(att1, att2):
                return Similarity.get(1.00)
            ancestor = self.taxonomy.get_common_ancestor(att1, att2)
            return self.taxonomy.get_similarity(ancestor)
        if config == TaxonomyConfig.INNER_NODES_PESSIMISTIC:
            ancestor = self.taxonomy.get_common_ancestor(att1, att2)
            return self.taxonomy.get_similarity(ancestor)
        if config == TaxonomyConfig.INNER_NODES_AVERAGE:
            if not self.taxonomy.contains_as_subnode(att1, att2):
                ancestor = self.taxonomy.get_common_ancestor(att1, att2)
                return self.taxonomy.get_similarity(ancestor)
            child_count = self.taxonomy.count_leaves(att1)
            sim = (1.0 + (child_count - 1) * self.taxonomy.get_similarity(att1).value) / child_count
            return Similarity.get(sim)
        return Similarity.INVALID_SIM

    def set_node_similarity(self, att: TaxonomyNode, sim: Similarity) -> None:
        self.taxonomy.set_similarity(att, sim)

    def set_query_config(self, query_config: TaxonomyConfig) -> None:
        if query_config != self.query_config and self.is_symmetric:
            self.case_config = query_config
        self.query_config = query_config
        self.update_table()

    def get_query_config(self) -> TaxonomyConfig:
        return self.query_config

    def set_case_config(self, case_config: TaxonomyConfig) -> None:
        if case_config != self.case_config:
            self.case_config = case_config
            if self.is_symmetric:
                self.query_config = case_config
            self.update_table()

    def get_case_config(self) -> TaxonomyConfig:
        return self.case_config

    def set_parent(self, symbol: TaxonomyNode, new_parent: TaxonomyNode) -> None:
        self.taxonomy.set_parent(symbol, new_parent)
        self.update_table()

    def get_common_ancestor(self, symbol_att1: SymbolAttribute, symbol_att2: SymbolAttribute) -> TaxonomyNode:
        return self.taxonomy.get_common_ancestor(symbol_att1, symbol_att2)

    def get_top_symbol(self) -> TaxonomyNode:
        return self.taxonomy.get_top_symbol()

    def get_similarity(self, node: TaxonomyNode) -> Similarity:
        return self.taxonomy.get_similarity(node)

    def get_parent(self, current_node: TaxonomyNode) -> TaxonomyNode | None:
        return self.taxonomy.get_parent(current_node)

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
        from hklii_psla.cbr.core.project import Project

        if type(new_desc) is type(self.desc) and self.name != Project.DEFAULT_FCT_NAME:
            assert isinstance(new_desc, SymbolDesc)
            f = new_desc.add_taxonomy_fct(self.name, is_active)
            f.sims = [row[:] for row in self.sims]
            f.is_symmetric = self.is_symmetric
            f.mc = self.mc
            f.case_config = self.case_config
            f.query_config = self.query_config
            f.taxonomy = self.taxonomy
