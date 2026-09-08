from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, final, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.sim_fct import SimFctInterface


class SimpleAttDesc(AttributeDesc, ABC):

    def __init__(
        self,
        owner: Concept,
        name: str
    ):
        super().__init__(owner, name)
        self._sim_fcts: dict[str, SimFctInterface] = {}

    @property
    def sim_fcts(self) -> list[SimFctInterface]:
        return list(self._sim_fcts.values())

    def get_fct(self, name: str) -> SimFctInterface | None:
        return self._sim_fcts.get(name)

    def add_function(self, f: SimFctInterface, active: bool) -> None:
        self._sim_fcts[f.name] = f
        if active:
            self._set_fct_active(f)

    def _set_fct_active(self, f: SimFctInterface) -> None:
        amalgam = self.owner.get_active_amalgam_fct()
        if amalgam is not None:
            amalgam.set_active_fct(self, f)

    def delete_sim_fct(self, f: SimFctInterface | None) -> None:
        if f is None:
            return
        self._sim_fcts.pop(f.name, None)
        if not self._sim_fcts:
            self.add_default_fct()

    def add_fct(self, f: SimFctInterface) -> None:
        self._sim_fcts[f.name] = f

    @abstractmethod
    def add_default_fct(self) -> None:
        ...

    @final
    def delete_all_fcts(self) -> None:
        self._sim_fcts.clear()

    def rename_fct(self, old_name: str, new_name: str) -> None:
        fct = self._sim_fcts.get(old_name)
        if fct:
            self._sim_fcts[new_name] = fct
            del self._sim_fcts[old_name]

    @override
    def get_representation(self) -> dict[str, Any]:
        ret = super().get_representation()
        sim_fcts_ret: dict[str, dict[str, Any]] = {}
        for key, sim_fct in self._sim_fcts.items():
            if sim_fct:
                sim_fcts_ret[key] = sim_fct.get_representation()
        ret["sim_fct"] = sim_fcts_ret
        return ret
