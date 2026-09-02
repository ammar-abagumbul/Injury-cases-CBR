from typing import override


from hklii_psla.cbr.core.explanation.symbol_desc import SymbolDesc
from hklii_psla.cbr.core.project import Project
from hklii_psla.cbr.core.similarity.sim_fct import SimFct


class SymbolFct(SimFct):


    def __init__(
        self,
        project: Project,
        desc: SymbolDesc,
        name: str
    ):
        super().__init__(project, desc, name)
        self._sub_desc = desc

        # TODO: count
