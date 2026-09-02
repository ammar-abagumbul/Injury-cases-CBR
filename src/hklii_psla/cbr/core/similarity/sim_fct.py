from abc import ABC, abstractmethod

from typing import Any, override

from hklii_psla.cbr.core.casebase.attribute import Attribute
from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.project import Project
from hklii_psla.cbr.core.similarity.similarity import Similarity

class SimFctInterface(ABC):

    @abstractmethod
    def calculate_similarity(
        self,
        a1: Attribute,
        a2: Attribute
    ) -> Similarity | None:
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

    @property
    @abstractmethod
    def project(self) -> Project | None:
        ...

    @property
    @abstractmethod
    def desc(self) -> AttributeDesc | None:
        ...

    @abstractmethod
    def clone(self, new_desc: AttributeDesc, is_active: bool):
        ...

    @abstractmethod
    def get_representation(self) -> dict[str, Any]:
        ...


class SimFct(SimFctInterface):

    _name: str
    _is_symetric: bool
    _project: Project | None
    _desc: AttributeDesc | None

    def __init__(
        self,
        project: Project,
        attr_desc: AttributeDesc,
        name: str
    ):
        self._name = name
        self._project = project
        self._desc = attr_desc

    @property
    @override
    def name(self) -> str:
        return self._name

    @name.setter
    @override
    def name(self, value: str):
        self._name = value

    @override
    def calculate_similarity(
        self,
        a1: Attribute,
        a2: Attribute
    ) -> Similarity | None:
        return None

    @property
    @override
    def is_symmetric(self) -> bool:
        return False

    @is_symmetric.setter
    @override
    def is_symmetric(self, value: bool) -> None:
        self._is_symetric = value

    @property
    @override
    def project(self) -> Project | None:
        return self._project

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        return None

    @property
    @override
    def desc(self) -> AttributeDesc | None:
        return None
