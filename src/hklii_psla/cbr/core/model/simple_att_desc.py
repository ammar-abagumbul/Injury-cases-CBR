from abc import ABC, abstractmethod

from typing import Any, final, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
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

    @final
    def delete_all_fcts(self) -> None:
        self._sim_fcts.clear()

    def rename_fct(self, old_name: str, new_name: str) -> None:
        fct = self._sim_fcts.get(old_name)
        if fct:
            self._sim_fcts[new_name] = fct
            del self._sim_fcts[old_name]
            # TODO: set_changed()
            # TODO: notify_observers()

    @override
    def get_representation(self) -> dict[str, Any]:
        ret = super().get_representation()
        sim_fcts_ret: dict[str, dict[str, Any]] = {}
        for key, sim_fct in self._sim_fcts.items():
            if sim_fct:
                sim_fcts_ret[key] = sim_fct.get_representation()
        ret["sim_fct"] = sim_fcts_ret
        return ret
