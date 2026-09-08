from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.project import Project


class ICaseBase(ABC):
    """A named collection of cases (instances) belonging to a project."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def set_name(self, name: str) -> None:
        ...

    @abstractmethod
    def contains_case(self, name: str) -> Instance | None:
        ...

    @abstractmethod
    def remove_case(self, name: str) -> bool:
        ...

    @abstractmethod
    def get_cases(self) -> list[Instance]:
        ...

    @abstractmethod
    def add_case(self, case: Instance) -> None:
        ...

    @property
    @abstractmethod
    def project(self) -> Project:
        ...
