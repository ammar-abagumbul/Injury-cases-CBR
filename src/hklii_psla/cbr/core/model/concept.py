from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.concept_range import ConceptRange
from hklii_psla.cbr.core.explanation.explainable import Explainable, ExplainableType
from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.project import Project


class Concept(Explainable):
    _id: str
    _project: Project | None
    _super_concept: Concept | None
    _range: ConceptRange | None
    _sub_concepts: dict[str, Concept]
    _attr_descs: dict[str, AttributeDesc]

    def __init__(
        self,
        ID: str,
        project: Project | None = None,
        super_concept: Concept | None = None,
    ) -> None:
        self._id = ID
        self._project = project
        self._super_concept = super_concept
        self._range = ConceptRange(project, self)

        if project is not None:
            if project.has_concept_with_id(ID):
                raise RuntimeError(f"Concept with name {ID} already exists")
            self._attr_desc = {}
            self._sub_concepts = {}
            if self._super_concept is None:
                raise RuntimeError(f"Concept with name {ID} doesn't have a super concept")
            self._super_concept.add_sub_concept(self, True)

    @property
    @override
    def name(self) -> str:
        return self._id

    @property
    @override
    def exp_type(self) -> ExplainableType:
        return ExplainableType.CONCEPT

    def add_sub_concept(self, c: Concept, is_new: bool) -> bool:
        if not is_new:
            temp: Concept | None = self
            while temp is not None and temp != self._project:
                if temp.name and temp.name == c.name:
                    return False
                temp = temp.super_concept

            if not self._check_overriden_attrs(self, c):
                return False

        self._sub_concepts[c.name] = c
        c.super_concept = self

    def _check_overriden_attrs(self, c1: Concept, c2: Concept):

        for a in c2.attr_descs.values():
            overriden_att = c1.attr_desc_by_name(a.name)

            if overriden_att and not a.can_override(overriden_att):
                return False

        for sub in c2.sub_concepts.values():
           if not self._check_overriden_attrs(c1, sub):
              return False

        return True

    @property
    def attr_descs(self):
        return self._attr_descs

    def attr_desc_by_name(self, name: str):
        return self._attr_descs.get(name)

    @property
    def super_concept(self):
        return self._super_concept

    @super_concept.setter
    def super_concept(self, value: Concept):
        self._super_concept = value

    @property
    def sub_concepts(self):
        return self._sub_concepts

    @property
    def all_sub_concepts(self) -> dict[str, Concept]:
        result: dict[str, Concept] = {}
        for key, value in self._sub_concepts.items():
            result[key] = value

        for sub in self._sub_concepts.values():
            result.update(sub._sub_concepts)

        return result
