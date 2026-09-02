from abc import ABC, abstractmethod
from enum import Enum


class ExplainableType(str, Enum):

    CONCEPT = "concept"
    ATTRIBUTE = "attribute"
    SIMPLE_ATTRIBUTE = "simple_attribute"
    INSTANCE = "instance"


class Explainable(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the instance."""
        ...

    @property
    @abstractmethod
    def exp_type(self) -> ExplainableType:
        """Returns the type of the instance."""
        ...
