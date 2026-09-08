from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from hklii_psla.cbr.core.model.concept import Concept

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
    from hklii_psla.cbr.core.default_case_base import DefaultCaseBase
    from hklii_psla.cbr.core.i_case_base import ICaseBase
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.special_desc import SpecialDesc
    from hklii_psla.cbr.core.similarity.similarity import Similarity
    from hklii_psla.cbr.core.similarity.special_fct import SpecialFct
    from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct
    from hklii_psla.cbr.core.similarity.taxonomy_fct import TaxonomyFct


class Project(Concept):
    """Root concept of a project's concept hierarchy: holds the vocabulary
    (concepts, attribute/similarity descriptions) and the case bases built
    from it. A Project has no super-concept and no parent project.
    """

    UNDEFINED_SPECIAL_ATTRIBUTE: ClassVar[str] = "_undefined_"
    UNKNOWN_SPECIAL_VALUE: ClassVar[str] = "_unknown_"
    NO_SPECIAL_VALUE: ClassVar[str] = "_others_"
    DEFAULT_FCT_NAME: ClassVar[str] = "default function"
    ID_DEFAULT: ClassVar[str] = "Project"

    def __init__(self, ID: str = ID_DEFAULT):
        super().__init__(ID, project=None, super_concept=None)
        self._case_bases: dict[str, ICaseBase] = {}
        self._special_value_desc: SpecialDesc = self._init_sv_desc()
        self._special_value_fct: SpecialFct
        self._inheritance_desc: SymbolDesc  # type: ignore[name-defined]
        self._inheritance_fct: TaxonomyFct
        self._init_fcts()

    # -- bootstrap ---------------------------------------------------------

    def _init_sv_desc(self) -> SpecialDesc:
        from hklii_psla.cbr.core.model.special_desc import SpecialDesc

        return SpecialDesc(
            self,
            {self.UNDEFINED_SPECIAL_ATTRIBUTE, self.UNKNOWN_SPECIAL_VALUE, self.NO_SPECIAL_VALUE},
        )

    def _init_fcts(self) -> None:
        from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
        from hklii_psla.cbr.core.similarity.config.taxonomy_config import TaxonomyConfig
        from hklii_psla.cbr.core.similarity.special_fct import SpecialFct

        self._special_value_fct = SpecialFct(self, self._special_value_desc, self.DEFAULT_FCT_NAME)
        self._special_value_desc.add_fct(self._special_value_fct)

        self._inheritance_desc = SymbolDesc(self, "inheritanceDesc", set())
        self._inheritance_fct = self._inheritance_desc.add_taxonomy_fct(self.DEFAULT_FCT_NAME, False)
        self._inheritance_fct.set_case_config(TaxonomyConfig.INNER_NODES_ANY)
        self._inheritance_fct.set_query_config(TaxonomyConfig.INNER_NODES_ANY)
        self._inheritance_fct.update_table()

    # -- Concept overrides: Project is the hierarchy root, not a modeled
    # concept itself, so it has no attributes, no super-concept, and no
    # part-of relations of its own. ------------------------------------

    @property
    @override
    def project(self) -> Project:
        return self

    @override
    def get_project(self) -> Project:
        return self

    @override
    def get_super_concept(self) -> Concept | None:
        return None

    @override
    def set_super_concept(self, c: Concept, is_new: bool) -> None:
        pass

    @property
    @override
    def attribute_descs(self) -> dict[str, AttributeDesc]:
        return {}

    @override
    def get_all_attribute_descs(self) -> dict[str, AttributeDesc]:
        return {}

    @override
    def add_attribute_desc(self, desc: AttributeDesc) -> None:
        pass

    @override
    def remove_attribute_desc(self, name: str) -> bool:
        return False

    @override
    def can_override(self, c: Concept) -> bool:
        return False

    @override
    def add_part_of_relation(self, desc) -> None:
        pass

    @override
    def get_part_of_relations(self) -> list:
        return []

    @override
    def delete_part_of_relation(self, desc) -> None:
        pass

    @override
    def delete(self) -> None:
        pass

    def has_concept_with_id(self, id_: str) -> bool:
        return id_ in self.get_all_sub_concepts()

    # -- case bases ----------------------------------------------------------

    def create_default_cb(self, name: str) -> DefaultCaseBase:
        from hklii_psla.cbr.core.default_case_base import DefaultCaseBase

        cb = DefaultCaseBase(self, name)
        self._case_bases[name] = cb
        return cb

    def has_cb(self, name: str) -> bool:
        return name in self._case_bases

    def get_cb(self, name: str) -> ICaseBase | None:
        return self._case_bases.get(name)

    def delete_case_base(self, name: str) -> ICaseBase | None:
        return self._case_bases.pop(name, None)

    def get_case_bases(self) -> dict[str, ICaseBase]:
        return self._case_bases

    def remove_case(self, name: str) -> None:
        for cb in self._case_bases.values():
            cb.remove_case(name)

    # -- concept hierarchy ----------------------------------------------

    def create_top_concept(self, id_: str) -> Concept:
        c = Concept(id_, self, self)
        self.sub_concepts[id_] = c
        return c

    def get_concept_by_id(self, id_: str) -> Concept | None:
        return self.get_all_sub_concepts().get(id_)

    @override
    def get_instance(self, name: str) -> Instance | None:
        for c in self.get_all_sub_concepts().values():
            i = c.get_instance(name)
            if i is not None:
                return i
        return None

    def get_all_instances(self) -> list[Instance]:
        result: list[Instance] = []
        for sub in self.sub_concepts.values():
            result.extend(sub.get_all_instances())
        return result

    # -- special values -------------------------------------------------

    def get_special_value_desc(self) -> SpecialDesc:
        return self._special_value_desc

    def is_special_attribute(self, obj: str | None) -> bool:
        if obj is None:
            return False
        if obj in (self.NO_SPECIAL_VALUE, self.UNDEFINED_SPECIAL_ATTRIBUTE, self.UNKNOWN_SPECIAL_VALUE):
            return True
        return self._special_value_desc.is_allowed_value(obj)

    def add_special_value(self, value: str) -> None:
        self._special_value_desc.add_symbol(value)

    def get_special_attribute(self, obj: str):
        return self._special_value_desc.get_attribute(obj)

    def get_special_fct(self) -> SymbolFct:
        return self._special_value_fct

    def set_special_value_fct(self, f: SpecialFct) -> None:
        self._special_value_fct = f

    # -- inheritance (used to compare instances of different concepts) --

    def get_inh_fct(self) -> TaxonomyFct:
        return self._inheritance_fct

    def add_inh_fct(self, name: str) -> TaxonomyFct:
        from hklii_psla.cbr.core.similarity.taxonomy_fct import TaxonomyFct

        f = TaxonomyFct(self, self._inheritance_desc, [], name)
        self._inheritance_desc.add_fct(f)
        return f

    # -- similarity helpers used across the similarity package ---------

    def calculate_special_similarity(self, att1: Attribute | None, att2: Attribute | None) -> Similarity:
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        no_special_value = self._special_value_desc.get_attribute(self.NO_SPECIAL_VALUE)
        if isinstance(att1, SpecialAttribute) and isinstance(att2, SpecialAttribute):
            return self._special_value_fct.calculate_special_similarity(att1, att2)
        if isinstance(att1, SpecialAttribute):
            return self._special_value_fct.calculate_special_similarity(att1, no_special_value)
        if isinstance(att2, SpecialAttribute):
            return self._special_value_fct.calculate_special_similarity(no_special_value, att2)
        return Similarity.INVALID_SIM

    def calculate_multiple_attribute_similarity(
        self, sim_fct: object, att1: MultipleAttribute | None, att2: MultipleAttribute | None
    ) -> Similarity:
        from hklii_psla.cbr.core.model.concept_desc import ConceptDesc
        from hklii_psla.cbr.core.similarity.amalgamation_fct import AmalgamationFct
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        if sim_fct is None or att1 is None or att2 is None:
            return Similarity.INVALID_SIM
        if not att1.attribute_desc.is_multiple:
            return Similarity.INVALID_SIM

        if isinstance(att1.attribute_desc, ConceptDesc) and isinstance(sim_fct, AmalgamationFct):
            mc = att1.attribute_desc.get_multiple_config(sim_fct)
            return mc.calculate_similarity(sim_fct, att1, att2)
        mc = getattr(sim_fct, "mc", None)
        if mc is not None:
            return mc.calculate_similarity(sim_fct, att1, att2)
        return Similarity.INVALID_SIM

    def clean_instances(self, c: Concept, desc: AttributeDesc) -> None:
        for i in list(c.get_direct_instances()):
            i.clean(desc)
        for sub in c.sub_concepts.values():
            self.clean_instances(sub, desc)
