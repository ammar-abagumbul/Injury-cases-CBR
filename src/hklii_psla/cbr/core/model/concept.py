from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.concept_range import ConceptRange
from hklii_psla.cbr.core.explanation.explainable import Explainable, ExplainableType
from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.similarity.config.amalgamation_config import AmalgamationConfig

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
    from hklii_psla.cbr.core.model.concept_desc import ConceptDesc
    from hklii_psla.cbr.core.project import Project
    from hklii_psla.cbr.core.similarity.amalgamation_fct import AmalgamationFct


class Concept(Explainable):

    def __init__(
        self,
        ID: str,
        project: Project | None = None,
        super_concept: Concept | None = None,
    ) -> None:
        if not ID or not ID.strip():
            raise ValueError("Cannot create concepts with an empty ID!")
        self._id = ID
        self._project: Project | None = project
        self._super_concept: Concept | None = super_concept
        self.range: ConceptRange = ConceptRange(project, self)
        self._sub_concepts: dict[str, Concept] = {}
        self._attr_descs: dict[str, AttributeDesc] = {}
        self._part_of_relations: list[ConceptDesc] = []
        self._available_amalgam_fcts: dict[str, AmalgamationFct] = {}
        self._active_amalgamation_fct: AmalgamationFct | None = None

        if project is not None:
            if project.has_concept_with_id(ID):
                raise ValueError(f'Concept with name "{ID}" already exists!')
            if self._super_concept is None:
                raise ValueError(f'Concept with name "{ID}" doesn\'t have a super concept')
            self._super_concept.add_sub_concept(self, True)

        from hklii_psla.cbr.core.project import Project as ProjectCls

        if not isinstance(self, ProjectCls):
            self.add_amalgamation_fct(AmalgamationConfig.EUCLIDEAN, ProjectCls.DEFAULT_FCT_NAME, True)

    # -- basic identity -------------------------------------------------

    @property
    @override
    def name(self) -> str:
        return self._id

    @property
    @override
    def exp_type(self) -> ExplainableType:
        return ExplainableType.CONCEPT

    @property
    def project(self) -> Project | None:
        return self._project

    def get_project(self) -> Project | None:
        return self._project

    def set_name(self, id_: str) -> None:
        if self._project is not None and not self._project.has_concept_with_id(id_):
            if self is not self._project and self._super_concept is not None:
                self._super_concept.rename_sub_concept(self._id, id_)
            self._id = id_
        else:
            raise ValueError(f'Concept with id "{id_}" already exists!')

    def __str__(self) -> str:
        return self._id

    # -- inheritance hierarchy -------------------------------------------

    @property
    def sub_concepts(self) -> dict[str, Concept]:
        return self._sub_concepts

    def get_all_sub_concepts(self) -> dict[str, Concept]:
        result: dict[str, Concept] = dict(self._sub_concepts)
        for sub in self._sub_concepts.values():
            result.update(sub.get_all_sub_concepts())
        return result

    def get_super_concept(self) -> Concept | None:
        return self._super_concept

    @property
    def super_concept(self) -> Concept | None:
        return self._super_concept

    @super_concept.setter
    def super_concept(self, value: Concept | None) -> None:
        self._super_concept = value

    def set_super_concept(self, c: Concept, is_new: bool) -> None:
        if self._super_concept is not c and c is not None:
            old_super_concept = self._super_concept
            c.add_sub_concept(self, is_new)
            self._super_concept = c
            if old_super_concept is not None:
                old_super_concept.remove_sub_concept(self._id)

    def remove_sub_concept(self, name: str) -> Concept | None:
        return self._sub_concepts.pop(name, None)

    def rename_sub_concept(self, name_old: str, name_new: str) -> None:
        sub = self._sub_concepts.pop(name_old, None)
        if sub is not None:
            self._sub_concepts[name_new] = sub

    def add_sub_concept(self, c: Concept, is_new: bool) -> bool:
        if c is None or c is self:
            return False

        if not is_new:
            tmp: Concept | None = self
            while tmp is not None and tmp is not self.project:
                if tmp.name == c.name:
                    return False
                tmp = tmp.get_super_concept()

            if not self._check_overridden_atts(self, c):
                return False

        self._sub_concepts[c.name] = c
        c.super_concept = self

        # keep the project's inheritance taxonomy (used for object-oriented
        # retrieval across concepts) in sync with the concept hierarchy.
        project = self.project
        if project is not None:
            inh_fct = project.get_inh_fct()
            inh_desc = inh_fct.desc
            parent: object
            if self is project:
                parent = inh_desc
            else:
                parent = inh_desc.get_attribute(self.name)
            if not is_new:
                inh_fct.taxonomy.parents[inh_desc.get_attribute(c.name)] = parent
            else:
                inh_desc.add_symbol(c.name)
                a = inh_desc.get_attribute(c.name)
                inh_fct.taxonomy.parents[a] = parent
                inh_fct.taxonomy.leaves.append(a)
                from hklii_psla.cbr.core.similarity.similarity import Similarity

                inh_fct.taxonomy.sims[a] = Similarity.get(1.00)

        return True

    def _check_overridden_atts(self, concept: Concept, c: Concept) -> bool:
        for a in c.attribute_descs.values():
            overridden_att = concept.get_attribute_desc(a.name)
            if overridden_att is not None and not a.can_override(overridden_att):
                return False
        for sub in c.sub_concepts.values():
            if not self._check_overridden_atts(concept, sub):
                return False
        return True

    def can_override(self, c: Concept) -> bool:
        return c.get_all_sub_concepts().get(self.name) is self or c is self

    # -- attribute descriptions -------------------------------------------

    @property
    def attribute_descs(self) -> dict[str, AttributeDesc]:
        return self._attr_descs

    def get_all_attribute_descs(self) -> dict[str, AttributeDesc]:
        result: dict[str, AttributeDesc] = {}
        path: list[Concept] = [self]
        tmp = self._super_concept
        while tmp is not None and tmp is not self._project:
            path.append(tmp)
            tmp = tmp.get_super_concept()
        for c in reversed(path):
            result.update(c.attribute_descs)
        return result

    def has_direct_attribute_desc(self, name: str) -> bool:
        return name in self._attr_descs

    def has_attribute_desc(self, name: str) -> bool:
        return name in self.get_all_attribute_descs()

    def get_attribute_desc(self, name: str) -> AttributeDesc | None:
        return self.get_all_attribute_descs().get(name)

    def get_attributes_of_sub_descs_for_name(self, name: str) -> list[AttributeDesc]:
        result: list[AttributeDesc] = []
        for sub in self._sub_concepts.values():
            att = sub.attribute_descs.get(name)
            if att is not None:
                result.append(att)
            else:
                result.extend(sub.get_attributes_of_sub_descs_for_name(name))
        return result

    def add_attribute_desc(self, desc: AttributeDesc) -> None:
        name = desc.name
        if self.has_direct_attribute_desc(name):
            raise ValueError(
                f'Cannot add attribute description. Concept "{self._id}" already '
                f'has an attribute description with name "{name}"'
            )

        desc_old = self.get_all_attribute_descs().get(name)
        override = False
        if desc_old is not None and desc is not None:
            if not desc.can_override(desc_old):
                raise ValueError(f'Cannot override the attribute with name "{name}" !')
            override = True

        sub_descs = self.get_attributes_of_sub_descs_for_name(name)
        for current_desc in sub_descs:
            if not current_desc.can_override(desc):
                raise ValueError(
                    f'Cannot add attribute! There is a naming conflict with an '
                    f'attribute called "{name}" in a sub concept!'
                )

        self._attr_descs[name] = desc

        if override:
            from hklii_psla.cbr.core.model.concept_desc import ConceptDesc
            from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc

            if not isinstance(desc, ConceptDesc) and isinstance(desc_old, SimpleAttDesc):
                for f in list(desc_old.sim_fcts):
                    f.clone(desc, False)

            for f in self.get_available_amalgam_fcts():
                weight = f.get_weight(desc_old)
                active = f.is_active(desc_old)
                active_fct = f.get_active_fct(desc_old)
                f.remove(desc_old)
                if active_fct is not None:
                    if not isinstance(desc, ConceptDesc) and isinstance(desc, SimpleAttDesc):
                        for current_fct in desc.sim_fcts:
                            if current_fct.name == getattr(active_fct, "name", None):
                                f.set_active_fct(desc, current_fct)
                                break
                    else:
                        f.set_active_fct(desc, active_fct)
                    if active is not None:
                        f.set_active(desc, active)
                if weight is not None:
                    f.set_weight(desc.name, weight)

    def remove_attribute_desc(self, name: str) -> bool:
        desc = self._attr_descs.pop(name, None)
        if desc is not None:
            from hklii_psla.cbr.core.model.concept_desc import ConceptDesc

            if isinstance(desc, ConceptDesc):
                try:
                    desc.concept.get_part_of_relations().remove(desc)
                except ValueError:
                    pass
            return True
        return False

    def rename_att_desc(self, name: str, name2: str) -> None:
        att = self._attr_descs.pop(name, None)
        if att is not None:
            self._attr_descs[name2] = att

    # -- instances ----------------------------------------------------------

    def get_direct_instances(self):
        return self.range.instances

    def remove_all_direct_instances(self) -> None:
        self.range.clear()

    def get_all_instances(self) -> list[Instance]:
        result = list(self.range.instances)
        for sub in self._sub_concepts.values():
            result.extend(sub.get_all_instances())
        return result

    def get_instance(self, name: str) -> Instance | None:
        res = self.range.contains(name)
        if res is None:
            for c in self._sub_concepts.values():
                res = c.get_instance(name)
                if res is not None:
                    return res
        return res

    def get_query_instance(self) -> Instance:
        from hklii_psla.cbr.core.casebase.instance import Instance

        query = Instance(self, "query")
        query.set_atts_unknown()
        return query

    def create_instance(self, name: str) -> Instance:
        i = self.range.contains(name)
        if i is None:
            i = self.range.get_instance(name)
        return i

    def add_instance(self, i: Instance) -> bool:
        return self.range.add(i)

    def copy_instance(self, original: Instance, new_name: str) -> Instance | None:
        i = self.range.contains(original.name)
        if i is None or i is not original:
            return None
        i = self.range.contains(new_name)
        if i is None:
            i = self.range.get_instance(new_name)
            for desc, value in original.attributes.items():
                i.add_attribute(desc, value)
        return i

    def rename_instance(self, name: str, name2: str) -> None:
        self.range.rename_instance(name, name2)

    def remove_instance(self, name: str) -> None:
        self.range.remove(name)

    def set_all_instances_multiple(self, desc: AttributeDesc) -> None:
        self.range.set_all_instances_multiple(desc)
        for c in self._sub_concepts.values():
            c.set_all_instances_multiple(desc)

    def set_all_instances_single(self, desc: AttributeDesc) -> None:
        self.range.set_all_instances_single(desc)
        for c in self._sub_concepts.values():
            c.set_all_instances_single(desc)

    # -- part-of relations ----------------------------------------------

    def add_part_of_relation(self, desc: ConceptDesc) -> None:
        self._part_of_relations.append(desc)

    def get_part_of_relations(self) -> list[ConceptDesc]:
        return self._part_of_relations

    def delete_part_of_relation(self, desc: ConceptDesc) -> None:
        try:
            self._part_of_relations.remove(desc)
        except ValueError:
            pass

    # -- amalgamation functions -------------------------------------------

    def get_fct(self, name: str) -> AmalgamationFct | None:
        return self._available_amalgam_fcts.get(name)

    def add_amalgamation_fct(self, amalgam: AmalgamationConfig, name: str, active: bool) -> AmalgamationFct:
        from hklii_psla.cbr.core.similarity.amalgamation_fct import AmalgamationFct

        f = AmalgamationFct(amalgam, self, name)
        self._available_amalgam_fcts[f.name] = f
        if active:
            self._active_amalgamation_fct = f
        return f

    def delete_amalgam_fct(self, f: AmalgamationFct | None) -> None:
        if f is None:
            return
        self._available_amalgam_fcts.pop(f.name, None)
        if self._active_amalgamation_fct is f:
            if not self._available_amalgam_fcts:
                from hklii_psla.cbr.core.project import Project as ProjectCls

                self.add_amalgamation_fct(AmalgamationConfig.WEIGHTED_SUM, ProjectCls.DEFAULT_FCT_NAME, True)
            self._active_amalgamation_fct = next(iter(self._available_amalgam_fcts.values()), None)

    def get_active_amalgam_fct(self) -> AmalgamationFct | None:
        return self._active_amalgamation_fct

    def set_active_amalgam_fct(self, amalgam: AmalgamationFct) -> None:
        if amalgam in self._available_amalgam_fcts.values():
            self._active_amalgamation_fct = amalgam

    def get_available_amalgam_fcts(self) -> list[AmalgamationFct]:
        return list(self._available_amalgam_fcts.values())

    def rename_amalgamation_fct(self, name_old: str, name_new: str) -> None:
        fct = self._available_amalgam_fcts.pop(name_old, None)
        if fct is not None:
            self._available_amalgam_fcts[name_new] = fct

    # -- deletion ---------------------------------------------------------

    def delete(self) -> None:
        from hklii_psla.cbr.core.model.string_desc import StringDesc

        self._available_amalgam_fcts.clear()
        self._active_amalgamation_fct = None

        for desc in list(self.get_part_of_relations()):
            n = desc.name
            o = desc.owner
            desc.delete()
            StringDesc(o, n)

        for desc in list(self.attribute_descs.values()):
            desc.delete()

        if self._super_concept is not None:
            self._super_concept.remove_sub_concept(self._id)
        if self._project is not None:
            self._project.get_inh_fct().desc.remove_symbol(self._id)
            for i in self.range.instances:
                for cb in self._project.get_case_bases().values():
                    cb.remove_case(i.name)

        self.range.clear()

        for sub in list(self._sub_concepts.values()):
            sub.delete()
