from __future__ import annotations

from abc import ABC, abstractmethod


class TaxonomyNode(ABC):

    @property
    @abstractmethod
    def nodes(self) -> list[TaxonomyNode]:
        ...
