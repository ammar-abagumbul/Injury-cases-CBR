from abc import ABC, abstractmethod


class Attribute(ABC):

    @abstractmethod
    def get_value_as_string(self) -> str:
        ...
