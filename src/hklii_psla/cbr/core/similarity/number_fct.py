from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.distance_config import DistanceConfig
from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig
from hklii_psla.cbr.core.similarity.sim_fct import SimFct

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc
    from hklii_psla.cbr.core.project import Project


class NumberFct(SimFct, ABC):
    """Computes similarity of integers, floats or doubles based on distance."""

    def __init__(self, p: Project, d: SimpleAttDesc, n: str):
        super().__init__(p, d, n)
        self.mc: MultipleConfig = MultipleConfig.DEFAULT_CONFIG
        self.max: float = 0.0
        self.min: float = 0.0
        self.diff: float = 0.0
        self.sub_desc = d
        self.distance_function: DistanceConfig = DistanceConfig.DIFFERENCE

    def get_multiple_config(self) -> MultipleConfig:
        return self.mc

    def set_multiple_config(self, mc: MultipleConfig) -> None:
        self.mc = mc

    @override
    def clone(self, new_desc, is_active) -> None:
        return None

    def set_distance_fct(self, df: DistanceConfig) -> None:
        if df != self.distance_function:
            if df == DistanceConfig.QUOTIENT and self.min <= 0 <= self.max:
                return
            self.distance_function = df

    def get_distance_fct(self) -> DistanceConfig:
        return self.distance_function
