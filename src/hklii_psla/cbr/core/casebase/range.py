from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

from hklii_psla.cbr.core.casebase.attribute import Attribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.project import Project


class Range(ABC):

    def __init__(self, project: Project | None):
        self._project = project

    @abstractmethod
    def get_attribute(self, obj: object) -> Attribute | None:
        ...

    @abstractmethod
    def parse_value(self, string: str):
        ...

    @property
    def project(self) -> Project | None:
        return self._project

    @project.setter
    @final
    def project(self, value: Project | None) -> None:
        self._project = value
