from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.attribute import Attribute
from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.project import Project


class SimFctInterface(ABC):

    @abstractmethod
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        ...

    @property
    @abstractmethod
    def is_symmetric(self) -> bool:
        ...

    @is_symmetric.setter
    @abstractmethod
    def is_symmetric(self, value: bool):
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @name.setter
    @abstractmethod
    def name(self, value: str) -> None:
        ...

    @abstractmethod
    def clone(self, new_desc: AttributeDesc, is_active: bool):
        ...


class SimFct(SimFctInterface):
    """Base for similarity functions. ``prj``/``desc`` are plain mutable
    attributes (not properties) since concrete functions routinely rebind
    them (e.g. when cloning onto a new description)."""

    def __init__(self, project: Project, attr_desc: AttributeDesc, name: str):
        self._name = name
        self.prj: Project = project
        self.desc: AttributeDesc = attr_desc
        self._is_symmetric: bool = False

    @property
    @override
    def name(self) -> str:
        return self._name

    @name.setter
    @override
    def name(self, value: str):
        self._name = value

    @override
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        return Similarity.INVALID_SIM

    @property
    @override
    def is_symmetric(self) -> bool:
        return self._is_symmetric

    @is_symmetric.setter
    @override
    def is_symmetric(self, value: bool) -> None:
        self._is_symmetric = value

    @property
    def project(self) -> Project:
        return self.prj

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        return None
